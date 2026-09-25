# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
coverage.py — per-pixel cloud/coverage flags for the AMW inference pipeline.

WHY THIS EXISTS (the "cloud flag")
----------------------------------
The upstream pipeline (core/gee.py, pinned at upstream/main c473933) builds a
median composite of Sentinel-2 scenes after per-scene cloud masking
(Cloud Score+ ``cs_cdf >= clear_threshold``). If a pixel is cloudy in EVERY
scene of the window, nothing survives the median and Earth Engine's
``computePixels`` fills the pixel with 0 in all bands. Downstream, an all-zero
chip simply scores "no mine" — a cloud gap is therefore INDISTINGUISHABLE from
a true negative in the detection output.

For an econometric time series this is fatal: apparent dips in mining area may
just be wet-season clouds (non-random missingness). This module produces the
missingness indicator the interpolation strategy needs, WITHOUT modifying any
upstream file (so `git checkout upstream/main -- core/` stays safe).

WHAT IT COMPUTES, per tile and date window
------------------------------------------
A 2-band raster on the exact same 576x576 grid as the imagery tile:

  band 1  N_CLEAR : number of scenes in the window in which this pixel passed
                    the cloud mask (cs_cdf >= clear_threshold). This is the
                    per-pixel sample size of the median composite — a direct
                    precision measure (more clear scenes -> less composite noise).
  band 2  N_TOTAL : number of scenes whose footprint covered this pixel at all
                    (before cloud masking). N_CLEAR/N_TOTAL is the clear-sky rate.

N_CLEAR == 0  <=>  the composite pixel is a fill value (gap), i.e. "missing",
which also lets us EMPIRICALLY VERIFY the fill=0 assumption flagged as
[UNVERIFIED] in pipeline_spatial_card.md: pixels with N_CLEAR == 0 should be
exactly the all-13-band-zero pixels of the imagery tile (see
``panel.fill_zero_mismatch``).

Results are cached as GeoTIFFs next to the imagery cache, keyed the same way
(collection_tileKey_start_end), so re-runs are free.
"""
# >> FILE OVERVIEW: fork-original (not in upstream Amazon Mining Watch).
# >> Makes the cloud "missingness" flag for the panel: per pixel, how many scenes were
# >> clear (N_CLEAR) and how many existed at all (N_TOTAL) in the time window.
# >> Same Earth Engine recipe as core/gee.py (S2L1C) but .count() on B2 instead of
# >> .median().
# >> Used by panel.py:build_period_panel (via _fetch_cov_safe and chip_coverage_stats).
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
# >> rasterio = GeoTIFF read/write library (used for the coverage cache files).
import rasterio
from rasterio.transform import from_bounds

# core/ modules (gee, tile_utils) must be importable; run_trials.py inserts
# <repo>/core onto sys.path before importing us. Direct users: see timeseries/README.md.
# >> ee = Earth Engine Python API; init reuses upstream's lazy, locked, run-once function.
import ee  # noqa: E402  (earthengine-api)
from gee import _ensure_earth_engine_initialized, TileType  # upstream core/gee.py
from tile_utils import ensure_tile_shape  # upstream core/tile_utils.py

log = logging.getLogger(__name__)

# >> Band names in output order: index 0 = N_CLEAR, index 1 = N_TOTAL.
BAND_NAMES = ["N_CLEAR", "N_TOTAL"]


# >> What: asks Earth Engine "how many usable images per pixel?" for each tile and window.
# >> Built once per period in panel.py:build_period_panel; get_tile_counts is called per
# >> tile, concurrently, from the panel's thread pool.
# >> Out per tile: (576, 576, 2) float32 counts, cached on disk as a uint16 GeoTIFF.
# >> Gotcha: clear_threshold is not in the cache filename (same as upstream's imagery
# >> cache), so changing the threshold would silently reuse old counts.
class CoverageExtractor:
    """Fetch per-pixel valid-observation counts for tiles, mirroring the
    composite construction in ``core/gee.GEE_Data_Extractor._build_composite``
    (S2L1C path) so that N_CLEAR counts exactly the scenes that entered the
    median composite the model saw.

    Parameters
    ----------
    start_date, end_date : str  (YYYY-MM-DD; end exclusive, as in core/gee.py)
    clear_threshold : float     MUST equal DataConfig.clear_threshold used for
                                the imagery pull (default 0.6), otherwise the
                                counts describe a different composite.
    cache_dir : Path | None     GeoTIFF cache directory (shared with imagery cache
                                is fine — file names cannot collide).
    """

    # >> Cache prefix S2COV keeps coverage files apart from imagery files (S2L1C_...).
    COLLECTION_TAG = "S2COV"  # cache-name prefix; distinct from imagery "S2L1C"

    # >> What: stores the window and cloud threshold, and prepares the server-side count
    # >> image. start/end: 'YYYY-MM-DD', end EXCLUSIVE. clear_threshold: Cloud Score+
    # >> cut-off, 0-1 (default 0.6). No pixels are downloaded here; EE images are lazy
    # >> recipes.
    def __init__(self, start_date: str, end_date: str,
                 clear_threshold: float = 0.6,
                 cache_dir: Optional[Path] = None):
        # >> Initialise Earth Engine once per process (EE_PROJECT env, high-volume
        # >> endpoint).
        _ensure_earth_engine_initialized()  # same lazy singleton as upstream
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.clear_threshold = float(clear_threshold)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # >> Create the cache folder if it is missing.
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.count_image = self._build_count_image()

    # ------------------------------------------------------------------ build
    # >> What: defines, on Google's servers, a 2-band image of scene counts per pixel.
    # >> Recipe: S2_HARMONIZED (Level-1C, top-of-atmosphere) -> filterDate (end exclusive)
    # >> -> link Cloud Score+ cs_cdf -> mask pixels with cs_cdf < threshold -> count() on
    # >> band B2.
    # >> Out: ee.Image with bands N_CLEAR, N_TOTAL as uint16; nothing runs until
    # >> computePixels.
    # >> Why count(), not median(): the median is the image the CNN saw; the count is how
    # >> many scenes went into that median at each pixel (its sample size, i.e.
    # >> precision).
    def _build_count_image(self) -> "ee.Image":
        """Server-side image with N_CLEAR and N_TOTAL bands.

        Mirrors core/gee.py S2L1C branch: COPERNICUS/S2_HARMONIZED linked with
        GOOGLE/CLOUD_SCORE_PLUS ``cs_cdf``, masked at ``clear_threshold``.
        ``ImageCollection.count()`` counts UNMASKED pixels per location, which is
        precisely "scenes that survived the cloud mask" (N_CLEAR) or "scenes
        covering the pixel" (N_TOTAL, no mask applied).

        We count on band B2 (native 10 m) so the count grid matches the 10 m
        request grid without resampling surprises.
        """
        # >> Cloud Score+ band cs_cdf: per-pixel clear-sky score, 0 = cloudy, 1 = clear.
        qa_band = "cs_cdf"
        # >> EE: Sentinel-2 L1C scenes dated in [start, end); the end date itself is
        # >> excluded.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(
            self.start_date, self.end_date)
        # >> EE: Cloud Score+ collection, one score image per Sentinel-2 scene.
        cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

        # >> EE: attach cs_cdf to each scene (matched by scene id), then per scene and per
        # >> pixel keep only pixels with cs_cdf >= clear_threshold (0.6). Same mask as the
        # >> imagery composite.
        masked = (
            s2.linkCollection(cs_plus, [qa_band])
            .map(lambda img: img.updateMask(
                img.select(qa_band).gte(self.clear_threshold)))
        )
        # >> EE count(): per pixel, number of scenes still unmasked on B2 (10 m blue
        # >> band).
        n_clear = masked.select("B2").count().rename("N_CLEAR")
        # >> N_TOTAL: same count WITHOUT the cloud mask = scenes whose footprint covers
        # >> the pixel.
        n_total = s2.select("B2").count().rename("N_TOTAL")

        # >> Stack as 2 bands; unmask(0) turns "no scenes at all" into an explicit 0;
        # >> toUint16 = store counts as unsigned 16-bit integers.
        # unmask(0): where zero scenes exist at all, count() is fully masked;
        # make that an explicit 0 instead of relying on the transport fill.
        return n_clear.addBands(n_total).unmask(0).toUint16()

    # ------------------------------------------------------------------ cache
    # >> What: the cache filename for one tile's counts, or None when caching is off.
    # >> Pattern: S2COV_<tile.key>_<start>_<end>.tif, e.g.
    # >> S2COV_552:12:10.0:19:-15:-263_2025-01-01_2025-02-01.tif (threshold not in the
    # >> name).
    def _cache_path(self, tile: TileType) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / (
            f"{self.COLLECTION_TAG}_{tile.key}_{self.start_date}_{self.end_date}.tif")

    # >> What: writes one tile's counts to disk so re-runs skip Earth Engine.
    # >> In: (576, 576, 2) array; converted to uint16 and to rasterio's (bands, H, W)
    # >> order.
    # >> FILE WRITE: GeoTIFF, EPSG:4326, deflate compression, internally tiled, band names
    # >> set.
    # >> Affine transform from the tile's lat/lon bounds, same convention as
    # >> core/gee.save_tile.
    def _save(self, arr: np.ndarray, tile: TileType, path: Path) -> None:
        """uint16 GeoTIFF, EPSG:4326, same affine convention as core/gee.save_tile."""
        # >> Reshape (H, W, B) = (576, 576, 2) -> (B, H, W) = (2, 576, 576) for rasterio.
        pixels = np.moveaxis(arr.astype("uint16", copy=False), -1, 0)  # (B,H,W)
        bands, height, width = pixels.shape
        profile = {
            "driver": "GTiff", "height": height, "width": width, "count": bands,
            "dtype": "uint16", "crs": "EPSG:4326",
            # >> from_bounds: maps the pixel grid onto the tile's lat/lon box (west,
            # >> south, east, north).
            "transform": from_bounds(*tile.geometry.bounds, width, height),
            "compress": "deflate", "tiled": True,
        }
        # >> FILE WRITE: the cache GeoTIFF (plus band descriptions N_CLEAR, N_TOTAL).
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(pixels)
            dst.descriptions = tuple(BAND_NAMES)

    # >> What: reads a cached counts GeoTIFF back into memory.
    # >> Out: (576, 576, 2) float32, from (2, 576, 576) uint16 on disk.
    @staticmethod
    def _load(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
        # >> Reshape (B, H, W) -> (H, W, B), the order panel.py asserts.
        return np.moveaxis(arr, 0, -1)  # (H,W,B)

    # ------------------------------------------------------------------ fetch
    # >> What: returns cloud counts for one tile: from the disk cache, else from Earth
    # >> Engine.
    # >> In: a DLTile tile (key, geometry in EPSG:4326, tilesize 552, pad 12).
    # >> Out: (576, 576, 2) float32 [N_CLEAR, N_TOTAL]; pixel (i, j) = imagery pixel (i,
    # >> j).
    # >> Called from panel.py:_fetch_cov_safe inside the thread pool.
    # >> Gotcha: unlike upstream get_tile_data there is no retry here; an EE error
    # >> propagates and
    # >> _fetch_cov_safe turns it into None (tile marked FAILED).
    def get_tile_counts(self, tile: TileType) -> np.ndarray:
        """Return (H, W, 2) float32 array [N_CLEAR, N_TOTAL] for one tile.

        Same request geometry as the imagery pull (clipToBoundsAndScale onto a
        (tilesize + 2*pad)^2 grid), so pixel (i, j) here corresponds to pixel
        (i, j) of the imagery tile. Cached on disk if cache_dir was given.
        """
        cache = self._cache_path(tile)
        # >> Cache hit: load and return; a corrupt file falls through to a fresh download.
        if cache and cache.exists():
            try:
                return self._load(cache)
            except Exception as e:  # corrupt cache -> refetch
                log.warning("Failed to load cached coverage %s: %s", cache, e)

        # >> 576 = 552 core + 2 x 12 px pad.
        out_size = tile.tilesize + 2 * tile.pad
        # >> EE rectangle from the tile's lat/lon bounds (EPSG:4326), same as the imagery
        # >> request.
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        # >> EE clipToBoundsAndScale: resample the counts onto a 576 x 576 grid over the
        # >> tile.
        expr = self.count_image.clipToBoundsAndScale(
            geometry=tile_geom, width=out_size, height=out_size)

        # >> EE computePixels: synchronous REST call; returns a NumPy structured array.
        raw = ee.data.computePixels({
            "bandIds": BAND_NAMES,
            "expression": expr,
            "fileFormat": "NUMPY_NDARRAY",
        })
        # >> Structured array (one named field per band) -> plain (576, 576, 2) array.
        if isinstance(raw, np.ndarray) and raw.dtype.fields is not None:
            arr = np.stack([raw[b] for b in BAND_NAMES], axis=-1)
        else:
            arr = np.array(raw)
        # >> ensure_tile_shape (upstream): crop or reflect-pad to exactly 576 x 576; then
        # >> float32.
        arr = ensure_tile_shape(arr, out_size).astype(np.float32, copy=False)

        # >> Save to the cache; a write failure is logged, not fatal.
        if cache:
            try:
                self._save(arr, tile, cache)
            except Exception as e:
                log.warning("Failed to cache coverage %s: %s", cache, e)
        return arr


# --------------------------------------------------------------------- helpers
# >> What: summarises cloud cover for one 48 x 48 px chip (4 panel columns).
# >> In: cov (576, 576, 2), chip top-left row/col, chip_size 48.
# >> Out: clear_frac (share of pixels with N_CLEAR > 0; 1 = fully seen, 0 = fully
# >> missing), n_clear_mean and n_clear_min (clear scenes per pixel), n_total_mean (all
# >> scenes).
# >> Called once per chip by panel.py:build_period_panel.
def chip_coverage_stats(cov: np.ndarray, row: int, col: int, chip_size: int
                        ) -> Tuple[float, float, float, float]:
    """Coverage summary for one chip window [row:row+cs, col:col+cs].

    Returns (clear_frac, n_clear_mean, n_clear_min, n_total_mean) where
    clear_frac = share of the chip's pixels with at least one clear scene
    (i.e. share NOT missing from the composite). This is the chip-level
    cloud flag: clear_frac < 1 means partially gap-filled, == 0 fully missing.
    """
    # >> Slice the chip window: (576, 576, 2) -> (48, 48, 2).
    win = cov[row:row + chip_size, col:col + chip_size]
    n_clear = win[..., 0]
    n_total = win[..., 1]
    return (
        float(np.mean(n_clear > 0)),
        float(np.mean(n_clear)),
        float(np.min(n_clear)),
        float(np.mean(n_total)),
    )
