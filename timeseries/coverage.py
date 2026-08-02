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
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# core/ modules (gee, tile_utils) must be importable; run_trials.py inserts
# <repo>/core onto sys.path before importing us. Direct users: see timeseries/README.md.
import ee  # noqa: E402  (earthengine-api)
from gee import _ensure_earth_engine_initialized, TileType  # upstream core/gee.py
from tile_utils import ensure_tile_shape  # upstream core/tile_utils.py

log = logging.getLogger(__name__)

BAND_NAMES = ["N_CLEAR", "N_TOTAL"]


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

    COLLECTION_TAG = "S2COV"  # cache-name prefix; distinct from imagery "S2L1C"

    def __init__(self, start_date: str, end_date: str,
                 clear_threshold: float = 0.6,
                 cache_dir: Optional[Path] = None):
        _ensure_earth_engine_initialized()  # same lazy singleton as upstream
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.clear_threshold = float(clear_threshold)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.count_image = self._build_count_image()

    # ------------------------------------------------------------------ build
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
        qa_band = "cs_cdf"
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(
            self.start_date, self.end_date)
        cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

        masked = (
            s2.linkCollection(cs_plus, [qa_band])
            .map(lambda img: img.updateMask(
                img.select(qa_band).gte(self.clear_threshold)))
        )
        n_clear = masked.select("B2").count().rename("N_CLEAR")
        n_total = s2.select("B2").count().rename("N_TOTAL")

        # unmask(0): where zero scenes exist at all, count() is fully masked;
        # make that an explicit 0 instead of relying on the transport fill.
        return n_clear.addBands(n_total).unmask(0).toUint16()

    # ------------------------------------------------------------------ cache
    def _cache_path(self, tile: TileType) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / (
            f"{self.COLLECTION_TAG}_{tile.key}_{self.start_date}_{self.end_date}.tif")

    def _save(self, arr: np.ndarray, tile: TileType, path: Path) -> None:
        """uint16 GeoTIFF, EPSG:4326, same affine convention as core/gee.save_tile."""
        pixels = np.moveaxis(arr.astype("uint16", copy=False), -1, 0)  # (B,H,W)
        bands, height, width = pixels.shape
        profile = {
            "driver": "GTiff", "height": height, "width": width, "count": bands,
            "dtype": "uint16", "crs": "EPSG:4326",
            "transform": from_bounds(*tile.geometry.bounds, width, height),
            "compress": "deflate", "tiled": True,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(pixels)
            dst.descriptions = tuple(BAND_NAMES)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
        return np.moveaxis(arr, 0, -1)  # (H,W,B)

    # ------------------------------------------------------------------ fetch
    def get_tile_counts(self, tile: TileType) -> np.ndarray:
        """Return (H, W, 2) float32 array [N_CLEAR, N_TOTAL] for one tile.

        Same request geometry as the imagery pull (clipToBoundsAndScale onto a
        (tilesize + 2*pad)^2 grid), so pixel (i, j) here corresponds to pixel
        (i, j) of the imagery tile. Cached on disk if cache_dir was given.
        """
        cache = self._cache_path(tile)
        if cache and cache.exists():
            try:
                return self._load(cache)
            except Exception as e:  # corrupt cache -> refetch
                log.warning("Failed to load cached coverage %s: %s", cache, e)

        out_size = tile.tilesize + 2 * tile.pad
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        expr = self.count_image.clipToBoundsAndScale(
            geometry=tile_geom, width=out_size, height=out_size)

        raw = ee.data.computePixels({
            "bandIds": BAND_NAMES,
            "expression": expr,
            "fileFormat": "NUMPY_NDARRAY",
        })
        if isinstance(raw, np.ndarray) and raw.dtype.fields is not None:
            arr = np.stack([raw[b] for b in BAND_NAMES], axis=-1)
        else:
            arr = np.array(raw)
        arr = ensure_tile_shape(arr, out_size).astype(np.float32, copy=False)

        if cache:
            try:
                self._save(arr, tile, cache)
            except Exception as e:
                log.warning("Failed to cache coverage %s: %s", cache, e)
        return arr


# --------------------------------------------------------------------- helpers
def chip_coverage_stats(cov: np.ndarray, row: int, col: int, chip_size: int
                        ) -> Tuple[float, float, float, float]:
    """Coverage summary for one chip window [row:row+cs, col:col+cs].

    Returns (clear_frac, n_clear_mean, n_clear_min, n_total_mean) where
    clear_frac = share of the chip's pixels with at least one clear scene
    (i.e. share NOT missing from the composite). This is the chip-level
    cloud flag: clear_frac < 1 means partially gap-filled, == 0 fully missing.
    """
    win = cov[row:row + chip_size, col:col + chip_size]
    n_clear = win[..., 0]
    n_total = win[..., 1]
    return (
        float(np.mean(n_clear > 0)),
        float(np.mean(n_clear)),
        float(np.min(n_clear)),
        float(np.mean(n_total)),
    )
