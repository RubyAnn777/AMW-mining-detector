# >> FILE OVERVIEW
# >> Talks to Google Earth Engine (EE): builds a cloud-masked satellite composite for a
# >> date window and downloads it tile by tile as numpy arrays; also reads/writes the
# >> GeoTIFF image cache and writes SAM2 mask/logit tiles.
# >> Pipeline position: step 3-4 (EE composite per tile -> download pixels).
# >> Upstream file (Amazon Mining Watch), byte-identical to upstream; no fork changes.
from dataclasses import dataclass
import logging
import os
import threading
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, ClassVar, Dict, Any

# >> ee = the Earth Engine Python client. Calls build a server-side recipe; nothing is
# >> computed until ee.data.computePixels asks for actual pixels.
import ee
# >> google.api_core.retry: generic Google retry decorator (used on get_tile_data).
from google.api_core import retry
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed

# >> descarteslabs is used ONLY for DLTile grid maths (no Descartes Labs API calls).
from descarteslabs.geo import DLTile
from tile_utils import CenteredTile, ensure_tile_shape

# >> A tile is either a DLTile (snapped to the global UTM grid) or a CenteredTile
# >> (centred on a point; see tile_utils.py). Both expose key/geometry/tilesize/pad.
TileType = Union[DLTile, CenteredTile]

# >> Google Cloud project billed for EE requests. Read from env EE_PROJECT; default
# >> 'earthindex' is upstream's project, so set EE_PROJECT to our own project.
EE_PROJECT = os.environ.get("EE_PROJECT", "earthindex")
# >> Module-level flag + lock: EE is initialised lazily, once per process.
_ee_initialized = False
_EE_INIT_LOCK = threading.Lock()


# >> Log in to Earth Engine once per process, the first time an extractor is created.
# >> Endpoint: https://earthengine-highvolume.googleapis.com (EE's high-volume API,
# >> meant for many parallel small requests like ours); project = EE_PROJECT.
# >> Auth: service-account JSON if GOOGLE_APPLICATION_CREDENTIALS is set, else the
# >> user credentials saved by `earthengine authenticate`.
# >> Double-checked locking (check, lock, check again) so 8 threads cannot init twice.
# >> Called by: GEE_Data_Extractor.__init__ (so importing gee.py does NOT log in).
def _ensure_earth_engine_initialized() -> None:
    """Lazily initialize the Earth Engine client (first GEE_Data_Extractor use).

    Thread-safe singleton: ``ee.Initialize`` runs at most once per process, even
    under concurrent first-time calls. This matters because ``ee.Initialize``
    internally fetches ``/algorithms`` and racing duplicate inits contribute to
    429 rate-limit responses.

    If ``GOOGLE_APPLICATION_CREDENTIALS`` is set to a path of a service account JSON
    file, uses :class:`google.oauth2.service_account.Credentials` with the Earth Engine
    scope. Otherwise uses the default client (e.g. user ``earthengine authenticate``).
    """
    global _ee_initialized
    # >> Fast path: already initialised, no lock needed.
    if _ee_initialized:
        return
    # >> Lock so only one thread runs ee.Initialize; the second check inside the lock
    # >> catches a thread that finished init while we were waiting.
    with _EE_INIT_LOCK:
        if _ee_initialized:
            return
        # >> Env var holding a path to a service-account key file (JSON); empty = not set.
        key_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        init_kw: Dict[str, Any] = {
            # >> High-volume endpoint (not the default interactive one).
            "opt_url": "https://earthengine-highvolume.googleapis.com",
            "project": EE_PROJECT,
        }
        # >> Service-account branch: fail loudly if the key file path does not exist.
        if key_path:
            if not Path(key_path).is_file():
                raise FileNotFoundError(
                    f"GOOGLE_APPLICATION_CREDENTIALS={key_path!r} is not a readable file"
                )
            from google.oauth2 import service_account

            # >> OAuth scope that grants Earth Engine access to the service account.
            ee_scopes = ["https://www.googleapis.com/auth/earthengine"]
            credentials = service_account.Credentials.from_service_account_file(
                key_path, scopes=ee_scopes
            )
            init_kw["credentials"] = credentials
        # >> EE CALL: ee.Initialize(opt_url=high-volume URL, project=..., [credentials]).
        # >> Network call; raises if auth is missing or the project lacks EE access.
        ee.Initialize(**init_kw)
        _ee_initialized = True


PathLike = Union[str, Path]

# >> Settings for WHAT imagery to fetch and HOW big each tile is (one per run).
# >> Defaults = production: 552 px tile + 12 px pad per side = 576x576 px at 10 m
# >> (~5.76 km incl. pad), Sentinel-2 L1C, 13 bands, Cloud Score+ clear >= 0.6.
# >> Used by: inference_pipeline.py (CLI flags mirror these fields), sam2_mask.py,
# >> time_tiles.py, utils/gee_data_pull.py, timeseries/*.
# >> Gotcha: tilesize/pad reach filenames via tile.key, but clear_threshold does NOT,
# >> so cached tiles from a different threshold are silently reused.
@dataclass
class DataConfig:
    # >> Tile core size in pixels (552 px x 10 m = 5.52 km). DLTile grid spacing.
    tilesize: int = 552
    # >> Extra pixels on each side (12 px = 120 m) so chips overlap neighbouring tiles.
    pad: int = 12
    # >> Which recipe/band list to use; key into _BAND_IDS below.
    collection: str = "S2L1C"
    # >> None -> filled from _BAND_IDS[collection] in __post_init__.
    bands: Optional[List[str]] = None
    # >> Cloud Score+ cs_cdf cutoff (0 = cloudy, 1 = clear). Pixels below it are masked
    # >> out of each scene before the median. The ONLY cloud mask in scoring.
    clear_threshold: float = 0.6
    # >> Thread count for get_tile_data_concurrent AND for bulk_predict's producers.
    max_workers: int = 8
    # >> Folder for the GeoTIFF image cache (None = no cache).
    image_cache_dir: Optional[PathLike] = None #If given, will write/read images

    # >> Band names requested from EE, in this exact order = array channel order.
    # >> S2L1C (13 bands) index: 0 B1 coastal 443nm 60m | 1 B2 blue 490nm 10m |
    # >> 2 B3 green 560nm 10m | 3 B4 red 665nm 10m | 4-6 B5/B6/B7 red-edge 705/740/783nm
    # >> 20m | 7 B8A narrow NIR 865nm 20m | 8 B8 NIR 842nm 10m | 9 B9 water vapour 945nm
    # >> 60m | 10 B10 cirrus 1375nm 60m | 11 B11 SWIR 1610nm 20m | 12 B12 SWIR 2190nm 20m.
    # >> Gotcha: B8A comes BEFORE B8 (index 7 = B8A, 8 = B8). The model expects this.
    # >> EE resamples every band to the requested ~10 m tile grid.
    _BAND_IDS: ClassVar[Dict[str, List[str]]] = {
        # >> Sentinel-1 radar: VV / VH polarisations (not used in production).
        "S1": ["VV", "VH"],
        # >> Production collection: Sentinel-2 Level-1C, top-of-atmosphere reflectance x
        # >> 10000.
        "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8A", "B8", "B9", "B10", "B11", "B12"],
        # >> Same as S2L1C minus B10 (cirrus): 12 bands. Not used in production.
        "S2L1C-12band": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                          "B8A", "B8", "B9", "B11", "B12"],
        # >> Surface reflectance (L2A) product has no B10, so 12 bands. Not used here.
        "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8A", "B8", "B9", "B11", "B12"],
        # >> Google's AlphaEarth annual embeddings: 64 learned features A00..A63 per
        # >> pixel.
        "AlphaEarth": [f"A{x:02d}" for x in range(64)],
    }

    # >> Band names for an NDVI (vegetation index) calc: red B4, NIR B8A (narrow NIR).
    # >> Not read anywhere in this file.
    _NDVI_BANDS: ClassVar[Dict[str, str]] = {
        "red": "B4",
        "nir": "B8A"
    }

    # >> Runs right after the dataclass is built: fill in bands and create cache folder.
    def __post_init__(self):
        # Automatically resolve bands if not provided
        if self.bands is None:
            if self.collection not in self._BAND_IDS:
                raise ValueError(f"No band mapping defined for collection "
                                 f"'{self.collection}'")
            self.bands = self._BAND_IDS[self.collection]

        # >> Convert the cache path to a Path and create the folder (mkdir -p).
        if self.image_cache_dir:
            self.image_cache_dir = Path(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    # >> List the collection names this module knows (used for the --collection choices).
    @classmethod
    def available_collections(cls) -> List[str]:
        """Return the list of supported collection IDs."""
        return list(cls._BAND_IDS.keys())
    

# >> The Earth Engine downloader: one instance = one date window + one DataConfig.
# >> Builds the composite recipe once in __init__, then fetches pixels per tile.
# >> Created by: InferenceEngine.__init__ (inference_engine.py), sam2_mask.py,
# >> utils/gee_data_pull.py; also used by the fork's timeseries/panel.py.
class GEE_Data_Extractor:
    # >> Log in to EE (lazy, once), remember the dates, and build the composite recipe.
    # >> start_date/end_date: 'YYYY-MM-DD' strings. end_date is EXCLUSIVE in EE.
    def __init__(self, start_date: str, end_date: str, config: DataConfig):
        # >> First use of EE in the process triggers ee.Initialize here.
        _ensure_earth_engine_initialized()
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.config = config
        self.logger = logging.getLogger(__name__)
        # >> Builds a lazy server-side ee.Image; no pixels are transferred yet.
        self.composite = self._build_composite(start_date, end_date)

    # >> Write the EE 'recipe' for one cloud-free picture of the whole date window.
    # >> Returns a lazy ee.Image covering the globe; get_tile_data later cuts tiles out.
    # >> S2 recipe: all scenes in window -> attach Cloud Score+ -> mask cloudy pixels in
    # >> each scene -> per-pixel median of what is left (median resists leftover haze).
    # >> Gotcha: filterDate's end is EXCLUSIVE, so '2024-12-31' drops Dec 31.
    # >> Gotcha: a pixel cloudy in every scene has no data; it downloads as 0 in all
    # >> bands.
    def _build_composite(self, start_date: str, end_date: str):
        collection = self.config.collection
        clear_threshold = self.config.clear_threshold
      
        # >> Production branch (collection='S2L1C').
        if collection in ['S2L1C', 'S2L1C-12band']:
            # >> EE CALL: Sentinel-2 Level-1C (top-of-atmosphere), 'harmonized' =
            # >> processing baseline offset removed so pre/post-2022 scenes are
            # >> comparable.
            s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            # >> EE CALL: Google Cloud Score+ collection, one QA image per S2 scene.
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            # >> cs_cdf = Cloud Score+ 'clear' probability-like score, 0 = cloudy, 1 =
            # >> clear.
            QA_BAND = "cs_cdf"
            composite = (
                # >> EE CALL: keep scenes with start <= date < end (end exclusive).
                s2.filterDate(start_date, end_date)
                # >> EE CALL: linkCollection joins each S2 scene to its Cloud Score+ image
                # >> (same system:index) and adds the cs_cdf band to it.
                .linkCollection(csPlus, [QA_BAND])
                # >> EE CALL: per scene, per pixel: keep pixel only if cs_cdf >=
                # >> clear_threshold (0.6).
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                # >> EE CALL: median over time of the unmasked pixels, per band -> one
                # >> ee.Image.
                # >> Band values stay reflectance x 10000 (median of an even count can be
                # >> fractional).
                .median())

        # >> Surface reflectance variant: same cloud recipe, different base collection.
        elif collection == 'S2L2A':
            # >> EE CALL: Sentinel-2 Level-2A (atmospherically corrected surface
            # >> reflectance).
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                s2.filterDate(start_date, end_date)
                .linkCollection(csPlus, [QA_BAND])
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                .median())

        # >> Radar variant (unused in production): no cloud mask needed for radar.
        elif collection == 'S1':
            # >> EE CALL: Sentinel-1 ground-range-detected radar scenes.
            s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
            composite = (
                s1.filterDate(start_date, end_date)
                # >> EE CALL: keep Interferometric Wide swath mode with both VV and VH
                # >> bands.
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VH"))
                # >> EE CALL: mosaic() = top image wins per pixel (no averaging, unlike
                # >> median).
                .mosaic())

        # >> AlphaEarth variant (unused in production): Google's annual embedding images.
        elif collection == 'AlphaEarth':
            emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            # >> EE CALL: filter to the window and mosaic the annual embedding tiles
            # >> together.
            composite = emb.filterDate(start_date, end_date).mosaic()

        else:
            raise ValueError(f'Collection {collection} not recognized.')
        return composite

    # >> Get one tile's pixels as a (576, 576, n_bands) float32 array; cache-aware.
    # >> 1) If a cached GeoTIFF exists, read it and return (no network).
    # >> 2) Else ask EE for the composite clipped to the tile bounds at 576x576 px
    # >>    via ee.data.computePixels (synchronous REST call, returns numpy).
    # >> 3) Write the cache GeoTIFF (uint16 for S2) for next time.
    # >> Cache name: {collection}_{tile.key}_{start}_{end}.tif, e.g.
    # >> S2L1C_552:12:10.0:19:-15:-263_2024-01-01_2024-12-31.tif (threshold NOT in name).
    # >> Called by: InferenceEngine.produce_tile_input (producer threads), sam2_mask.py,
    # >> gee_data_pull.py, time_tiles.py, timeseries/panel.py.
    # >> Retry: google.api_core Retry, keeps retrying for up to 240 s total. Gotcha: its
    # >> default predicate only retries Google API 'transient' exception types (429/500/
    # >> 503 api_core errors); errors raised as ee.EEException are probably NOT retried.
    # >> Gotcha: first fetch returns float32 median values; a cache hit returns the
    # >> uint16-truncated copy, so fresh vs cached pixels can differ by < 1 unit.
    @retry.Retry(timeout=240)
    def get_tile_data(self, tile: TileType) -> np.ndarray:
        """Download or load cached image data for a tile.

        Inputs:
        - tile: a TileType object
        Outputs:
        - pixels: a numpy array with shape (H, W, bands)
        """
        image_cache_dir = getattr(self.config, "image_cache_dir", None)
        collection = self.config.collection
        start, end = self.start_date, self.end_date
        # >> Cache filename = collection + tile key (tilesize:pad:res:zone:x:y) + dates.
        tif_name = f"{collection}_{tile.key}_{start}_{end}.tif"

        # >> Cache hit path: read the GeoTIFF instead of calling EE.
        if image_cache_dir:
            tif_path = image_cache_dir / tif_name
            if tif_path.exists():
                try:
                    # >> FILE READ: load_tile returns (H, W, B) float32.
                    return self.load_tile(tif_path)
                except Exception as e:
                    # >> Corrupt/partial cache file: warn and fall through to a fresh
                    # >> download.
                    self.logger.warning(
                        f"Failed to load cached tile {tile.key}: {e}")

        # >> EE CALL: rectangle from the tile's lon/lat bounds (EPSG:4326, degrees).
        # >> CRS: tile.geometry is the UTM square reprojected to lon/lat; its bounding box
        # >> is slightly larger than the UTM square because the reprojected square is
        # >> tilted.
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        # >> Output pixels per side: 552 + 2*12 = 576.
        out_size = tile.tilesize + 2 * tile.pad
        # >> EE CALL: clipToBoundsAndScale resamples the composite onto a width x height
        # >> grid that exactly fills the lon/lat bounds -> a lat/lon pixel grid (~10 m,
        # >> not exactly square in metres). Same bounds feed affine_from_tile, so
        # >> georeferencing matches.
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom,
            width=out_size,
            height=out_size
        )

        # >> EE CALL: ee.data.computePixels = synchronous REST request that computes the
        # >> recipe server-side and returns the pixels directly (no export job).
        # >> NUMPY_NDARRAY -> a structured numpy array, one named field per band.
        pixels = ee.data.computePixels(
            {
                # >> Requested band list; the order here defines the channel order we
                # >> stack below.
                "bandIds": self.config.bands,
                "expression": composite_tile,
                "fileFormat": "NUMPY_NDARRAY",
            }
        )

        # >> Normalise the response into a plain (H, W, B) array.
        if isinstance(pixels, np.ndarray):
            # >> Structured array (fields B1, B2, ...): stack fields along a new last
            # >> axis.
            # >> Shape: (H, W) with 13 named fields -> (H, W, 13) in self.config.bands
            # >> order.
            if pixels.dtype.fields is not None:
                pixels = np.stack([pixels[band] for band in self.config.bands],
                                 axis=-1)
        else:
            # >> Anything else (e.g. list): coerce to an ndarray.
            pixels = np.array(pixels)

        # >> Force exactly 576x576: centre-crop if EE returned extra rows/cols,
        # >> reflect-pad if short (off-by-one sizes can happen at the grid edges).
        # >> tile_utils.py.
        pixels = ensure_tile_shape(pixels, out_size)
        # >> dtype conversion: to float32 for the model (copy=False avoids a copy if
        # >> already).
        pixels = pixels.astype(np.float32, copy=False)

        # >> Write-through cache: save the fresh download so the next run skips EE.
        if image_cache_dir:
            try:
                # >> FILE WRITE: GeoTIFF via save_tile (uint16 for S2, deflate,
                # >> EPSG:4326).
                self.save_tile(pixels, tile, image_cache_dir)
            except Exception as e:
                # >> Cache write failure is only a warning; the pixels are still returned.
                self.logger.warning(f"Failed to cache tile {tile.key}: {e}")

        return pixels

    # >> Download many tiles in parallel; returns a list aligned with `tiles` (None =
    # >> fail).
    # >> ThreadPoolExecutor with max_workers (8) threads; threads suit network waiting.
    # >> Two passes: failed tiles get one more concurrent pass, then give up with None.
    # >> Not used by bulk_predict (which has its own producer pool); a helper for others.
    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> List[Optional[np.ndarray]]:
        """
        Download all tile data concurrently.

        Per-tile failures do not abort the whole batch: failed indices are
        collected and retried in a second concurrent pass (helps transient EE
        errors such as user memory limit). Output order matches ``tiles``.
        Entries that still fail after retry are ``None``; an error is logged.
        """
        log = logging.getLogger(__name__)
        if not tiles:
            return []

        n = len(tiles)
        # >> Pre-sized result list so each thread writes its own slot (keeps input order).
        results: List[Optional[np.ndarray]] = [None] * n

        # >> Run get_tile_data for the given tile indices concurrently; return failed
        # >> ones.
        def run_indices(indices: List[int]) -> List[int]:
            failed_local: List[int] = []
            if not indices:
                return failed_local
            # >> THREAD POOL: max_workers threads, each calling get_tile_data (network
            # >> I/O).
            with ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                # >> Submit every tile; map future -> index so results land in the right
                # >> slot.
                future_to_i = {
                    executor.submit(self.get_tile_data, tiles[i]): i
                    for i in indices
                }
                # >> Collect results as they finish (completion order, not input order).
                for fut in as_completed(future_to_i):
                    i = future_to_i[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as exc:
                        log.warning(
                            "get_tile_data failed for tile %s: %s",
                            tiles[i].key,
                            exc,
                        )
                        failed_local.append(i)
            return failed_local

        # >> Pass 1 over all tiles.
        failed = run_indices(list(range(n)))
        if failed:
            log.info(
                "Retrying %d failed tile(s) after initial concurrent batch",
                len(failed),
            )
            # >> RETRY: pass 2 only for tiles that failed in pass 1.
            failed = run_indices(failed)
        if failed:
            sample = [tiles[i].key for i in failed[:5]]
            log.error(
                "get_tile_data still failed for %d tile(s) after retry "
                "(entries will be None). Sample keys: %r",
                len(failed),
                sample,
            )

        return list(results)

    # >> Build the pixel->map transform (affine) for a tile in EPSG:4326 lon/lat.
    # >> from_bounds(west, south, east, north, width, height): pixel (0, 0) = north-west.
    # >> Used by save_tile and SAM2_Masker.predict (inference_engine.py) to place masks.
    @staticmethod
    def affine_from_tile(tile, width: int, height: int):
        """Construct an EPSG:4326 affine transform from tile metadata."""
        return from_bounds(*tile.geometry.bounds, width, height)

    # >> Write a tile array to a GeoTIFF in EPSG:4326 and return its path.
    # >> Input: pixels (H, W, B); written as (B, H, W). Three kinds:
    # >>   image  -> uint16 for S2 (float32 otherwise), no nodata, {base}.tif.
    # >>   mask   -> uint8, nodata 2, {base}-msk.tif (SAM2 masks).
    # >>   logits -> float32, nodata NaN, {base}-logits.tif (SAM2 log-odds).
    # >> base = {collection}_{tile.key}_{start}_{end}; deflate, internally tiled.
    # >> Callers: get_tile_data (image cache), SAM2_Masker.predict, gee_data_pull.py.
    # >> Gotcha: float->uint16 cast truncates decimals and would wrap negatives/NaN.
    def save_tile(
        self, pixels: np.ndarray, tile: TileType, outdir: Path,
        product_type: Literal["image", "mask", "logits"] = "image") -> Path:
        """
        Write a tile GeoTIFF in EPSG:4326.

        product_type:
            - "image"  -> satellite imagery
            - "mask"   -> uint8 segmentation mask 
            - "logits" -> float32 model logits 
        """
        # >> SAM2 mask: 0/1 values, 2 = nodata.
        if product_type == "mask":
            dtype = "uint8"
            nodata = 2
            suffix = "-msk"
        # >> SAM2 logits (log-odds scores; > 0 means more likely mine), NaN = nodata.
        elif product_type == "logits":
            dtype = "float32"
            nodata = np.nan
            suffix = "-logits"
        else:  
            # >> Sentinel-2 imagery: reflectance x 10000 fits in uint16 (0..65535).
            if self.config.collection[:2] == "S2":
                dtype = "uint16"
            else:
                dtype = "float32"
            nodata = None
            suffix = ""

        # >> dtype conversion then axis move: (H, W, B) -> (B, H, W), rasterio's
        # >> band-first order.
        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        # >> Georeference: affine from the tile's lon/lat bounds (see affine_from_tile).
        transform = self.affine_from_tile(tile, width, height)
        # >> CRS: outputs are always EPSG:4326 (lon/lat), not UTM.
        crs = "EPSG:4326"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            # >> Compression: DEFLATE (lossless); tiled=True -> internal 256x256 blocks.
            "compress": "deflate",
            "tiled": True,
        }

        if nodata is not None:
            profile["nodata"] = nodata

        base = (
            f"{self.config.collection}_{tile.key}_"
            f"{self.start_date}_{self.end_date}"
        )

        tif_name = f"{base}{suffix}.tif"
        outpath = outdir / tif_name

        # >> FILE WRITE: rasterio writes the GeoTIFF (overwrites any existing file).
        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(pixels)

        return outpath


    # >> Read a cached GeoTIFF back into a (H, W, B) float32 array.
    # >> Called by get_tile_data on a cache hit. No shape check is done here.
    def load_tile(self, path: Path) -> np.ndarray:
        """Load a tile’s pixels from GeoTIFF."""
        with rasterio.open(path) as src:
            # >> FILE READ + dtype: uint16 on disk -> float32; array comes out (B, H, W).
            arr = src.read().astype(np.float32)  # (B,H,W)
            # >> Reshape: (B, H, W) -> (H, W, B), the layout the chipper and model expect.
            arr = np.moveaxis(arr, 0, -1)  # back to (H,W,B)
        return arr
