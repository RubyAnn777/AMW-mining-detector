# =============================================================================
# FILE OVERVIEW: gee.py
#
# PURPOSE (the heart of the pipeline)
#   This module implements the full artisanal-gold-mine detection pipeline
#   on top of Google Earth Engine (GEE) imagery:
#     1. GEE_Data_Extractor builds a LAZY per-pixel median/mosaic composite
#        recipe for a region+date-range+collection (e.g. cloud-masked
#        Sentinel-2 L1C) and downloads it, tile by tile, as numpy arrays
#        via ee.data.computePixels (synchronous, no export-to-Drive step).
#     2. InferenceEngine turns raw tile pixels into overlapping 48x48-px
#        "chips" (~23 ha each), embeds each chip with a frozen SSL4EO
#        ViT-S/16 backbone (PyTorch, 384-d output), and classifies the
#        embeddings with a Keras ensemble of small MLP heads (mean of 6
#        sigmoid outputs = "confidence"). Chips above `pred_threshold` are
#        kept as detections and written out as a GeoJSON of squares.
#     3. SAM2_Masker optionally (InferenceConfig.run_sam2) turns detection
#        chip geometries into a single box prompt per tile, runs a
#        fine-tuned SAM2 (Segment Anything 2) image segmentation model, and
#        writes a pixel-precise binary mask + raw logits GeoTIFF per tile —
#        used for area estimation of the actual mine scar, not just the
#        (much coarser) 48x48-px chip grid.
#   Region geometry -> DLTile/CenteredTile grid (tile_utils.create_tiles)
#   -> GEE composite fetch -> chip cut -> embed -> classify -> (opt.) SAM2
#   -> GeoJSON on disk. See inference_pipeline.py for the CLI entry point
#   that drives InferenceEngine.bulk_predict over an entire region.
#
# INVOCATION
#   Not run directly; imported by inference_pipeline.py, sam2_mask.py, and
#   the various notebooks (inference.ipynb, cloud_mask_filter.ipynb, etc).
#   IMPORTANT: importing this module calls ee.Initialize(...) at import
#   time (see below), which requires Google Earth Engine credentials to
#   already be configured (Application Default Credentials, e.g. via
#   `gcloud auth application-default login`) and a valid EE_PROJECT.
#
# ON-DISK INPUTS
#   - models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt: pickled
#     PyTorch ViT-S/16 backbone (in_chans=13, outputs 384-d embeddings).
#   - A Keras .h5 ensemble classifier (InferenceConfig.model_path).
#   - models/sam2/: cloned SAM2 repo (Hydra configs + base checkpoints)
#     plus a fine-tuned checkpoint (MaskConfig.finetuned_weights), used
#     only when SAM2 segmentation is enabled.
#   - Optional on-disk caches: per-tile GeoTIFFs (DataConfig.image_cache_dir)
#     and per-tile embeddings parquet files
#     (InferenceConfig.embeddings_cache_dir).
#
# ON-DISK OUTPUTS
#   - Cumulative positive-chip detections GeoJSON at
#     data/outputs/<model_version>/<region>_<model_version>_<thr>_
#     <start>_<end>.geojson (see InferenceEngine.predictions_geojson_path
#     and bulk_predict).
#   - Optional SAM2 mask/logits GeoTIFFs under MaskConfig.mask_dir.
#
# PIPELINE POSITION
#   This is the computational core invoked by inference_pipeline.py (bulk
#   CLI runs) and by sam2_mask.py / notebooks (interactive or masking-only
#   runs). tile_utils.py (annotated separately) supplies the tiling grid
#   and chip-cutting utilities this module calls into.
# =============================================================================

# ThreadPoolExecutor/as_completed power the producer/consumer concurrency
# in InferenceEngine.bulk_predict (parallel GEE downloads feeding a single
# serialized GPU/TF consumer thread).
from concurrent.futures import ThreadPoolExecutor, as_completed
# dataclass decorator used for DataConfig/InferenceConfig/MaskConfig.
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import queue
import threading
# ClassVar marks the per-collection band-name lookup dicts as class-level
# (not per-instance dataclass fields); the rest are standard typing helpers.
from typing import List, Literal, Optional, Tuple, Union, ClassVar, Dict, Any
import warnings

# Affine: 2D affine transform type used for pixel<->geographic coordinate
# mapping (rasterio/GDAL convention).
from affine import Affine
# DLTile: Descartes Labs UTM tiling-grid tile object (one of the two
# concrete TileType implementations; the other, CenteredTile, lives in
# tile_utils.py).
from descarteslabs.geo import DLTile
# Google Earth Engine Python client.
import ee
import geopandas as gpd
# google-api-core retry decorator, used to retry transient GEE errors.
from google.api_core import retry
import numpy as np
import pandas as pd
import rasterio
# from_bounds: builds a rasterio Affine transform from geographic bounds
# plus pixel width/height (used to georeference downloaded tiles).
from rasterio.transform import from_bounds
# zonal_stats: imported but not otherwise referenced in this module at
# time of writing (kept in case downstream consumers rely on the import,
# or historical dead import from a removed code path).
from rasterstats import zonal_stats
# scipy.ndimage: distance transforms, zoom (resampling), and Gaussian
# smoothing used by SAM2_Masker's spatial-prior/upsampling logic.
import scipy.ndimage as ndi
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local tiling/chip-cutting helpers (see tile_utils.py):
#   CenteredTile: a lightweight non-DLTile tile type.
#   cut_chips: slices a tile's pixel array into overlapping chips + geoms.
#   create_tiles: covers a region geometry with a grid of tiles.
#   ensure_tile_shape: pads/crops a downloaded array to the expected size.
from tile_utils import CenteredTile, cut_chips, create_tiles, ensure_tile_shape

# Union type covering both concrete tile implementations used throughout
# this module's type hints (DLTile from descarteslabs, CenteredTile local).
TileType = Union[DLTile, CenteredTile]
# Convenience alias: accept either a str or a pathlib.Path wherever a
# filesystem path is expected (dataclass fields normalize these in
# __post_init__).
PathLike = Union[str, Path]

# Repository root (directory that contains ``gee/`` and ``models/``).
# Derived from this file's location so paths work regardless of the
# process's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Absolute path to the pickled SSL4EO ViT-S/16 backbone weights (PyTorch
# full-module pickle, in_chans=13 matching the 13-band S2L1C stack,
# outputs 384-d embeddings). Used as the default InferenceConfig
# embed_model_path.
SSL4EO_PATH = str(
    (REPO_ROOT / "models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt").resolve()
)

# Absolute path to the cloned SAM2 repository (contains Hydra configs and
# base checkpoints); default MaskConfig.sam2_repo_path.
SAM2_PATH = str((REPO_ROOT / "models/sam2").resolve())
# Default output directory for SAM2 mask/logits GeoTIFFs.
DEFAULT_MASK_DIR = str((REPO_ROOT / "data/outputs/sam2").resolve())
# Default base directory for bulk-inference prediction GeoJSONs (a
# per-model-version subfolder is created under this at runtime).
DEFAULT_INFERENCE_OUTPUT_BASE = str((REPO_ROOT / "data/outputs").resolve())

# SAM2 ``build_sam2`` uses Hydra ``compose(config_name=...)`` — this must be a
# config name relative to the installed ``sam2`` package (see ``sam2.build_sam``),
# not an absolute filesystem path. Passing ``/Users/.../sam2.1_hiera_s.yaml``
# makes Hydra look for a config literally named ``Users/...`` (leading ``/`` lost).
DEFAULT_SAM2_HYDRA_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Google Cloud project used for Earth Engine API calls. Defaults to
# Earth Genome's own project ('earthindex'); other users of this fork
# MUST export their own EE_PROJECT env var pointing at a project they
# have Earth Engine access to, or GEE calls below will fail/be billed
# to the wrong account.
EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')
# MODULE-IMPORT-TIME SIDE EFFECT: this call runs as soon as `gee` is
# imported (not lazily), so importing this module requires valid Earth
# Engine credentials (Application Default Credentials) to already be set
# up on the machine. The "highvolume" endpoint is a special EE API
# endpoint designed to serve many concurrent programmatic requests (as
# opposed to the default endpoint, tuned for interactive/notebook use) —
# needed here because InferenceEngine.bulk_predict issues many concurrent
# per-tile computePixels calls.
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com",
              project=EE_PROJECT)

# Mac-only TensorFlow workaround: on Darwin (macOS), force TF to run ops
# eagerly and enable tf.data debug mode. This sidesteps a known TF/macOS
# bug with graph-mode execution; it is slower than graph mode, but the
# Keras MLP ensemble heads used here are tiny, so the perf cost is
# negligible in practice.
if platform.system() == 'Darwin':
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

@dataclass
class DataConfig:
    """Configuration for GEE image acquisition and tiling.

    Fields:
        tilesize: int, pixels per side of a downloaded tile request (no
            padding). Default 576 px; at ~10 m/px this is a 5.76 km
            square (~33.2 km^2) tile on the Descartes Labs DLTile UTM
            grid. This is the unit of work for GEE downloads (one
            ee.data.computePixels call per tile).
        pad: int, extra context pixels added to EACH side of the tile
            when requesting imagery (so the actual downloaded raster is
            (tilesize + 2*pad) px square). Default 0 (no padding).
        collection: str, which GEE image collection/recipe to build; must
            be a key of DataConfig._BAND_IDS (e.g. "S2L1C", "S2L2A",
            "S1", "AlphaEarth"). Default "S2L1C".
        bands: Optional[List[str]], explicit band IDs to request from GEE,
            in order. If None (default), auto-resolved in __post_init__
            from DataConfig._BAND_IDS[collection]. Band order matters: it
            must match the order the SSL4EO backbone was pretrained on.
        clear_threshold: float in [0,1], the Cloud Score+ `cs_cdf`
            (cloud-free confidence) threshold a pixel-observation must
            meet or exceed to be kept in the per-scene mask before
            compositing (S2 only). Higher = stricter cloud rejection,
            i.e. fewer surviving observations feed the median composite.
            Default 0.6.
        max_workers: int, number of concurrent threads used for GEE tile
            downloads (both get_tile_data_concurrent and the producer
            pool in InferenceEngine.bulk_predict). Default 8.
        image_cache_dir: Optional[str|Path], if given, downloaded tile
            rasters are cached to/read from GeoTIFFs in this directory
            (created if it does not already exist); avoids re-downloading
            identical tile/date/collection combinations across runs.
            Default None (no caching).

    Class-level lookup tables (ClassVar, shared across instances, not
    dataclass fields / not part of __init__):
        _BAND_IDS: maps a collection name to its ordered list of GEE band
            IDs to request. ORDER MATTERS for S2L1C: it must match the
            channel order the SSL4EO ViT backbone was pretrained on
            (note B8A is listed before B8, not in numeric order).
            S2L1C-12band and S2L2A both drop B10 (the cirrus band, which
            is absent from Level-2A surface-reflectance products). S1 is
            radar VV/VH polarizations. AlphaEarth is Google's 64-band
            (A00..A63) annual satellite embedding product.
        _NDVI_BANDS: {"red": "B4", "nir": "B8A"} — a leftover of a removed
            NDVI-based masking stage; still referenced (dead branch) by
            postprocess.py's --ndvi_threshold option.

    Side effects (in __post_init__): resolves `bands` from `collection`
    if not explicitly given; converts `image_cache_dir` to a Path and
    creates it on disk if set.
    """
    # Pixels per side of a GEE tile request (no padding). ~10 m/px, so
    # 576 px ~= 5.76 km per side (~33.2 km^2 per tile).
    tilesize: int = 576
    # Extra context pixels added per side beyond `tilesize` when fetching.
    pad: int = 0
    # Which GEE collection/recipe to use; must be a key of _BAND_IDS.
    collection: str = "S2L1C"
    # Explicit band list; None => auto-derived from _BAND_IDS[collection].
    bands: Optional[List[str]] = None
    # Cloud Score+ cs_cdf threshold: per-scene pixels below this are
    # masked out before the median composite (S2 collections only).
    clear_threshold: float = 0.6
    # Concurrent GEE download threads.
    max_workers: int = 8
    image_cache_dir: Optional[PathLike] = None #If given, will write/read images

    # Per-collection ordered band-ID lists. Order matters (must match the
    # channel order the embedding backbone expects / was trained on).
    _BAND_IDS: ClassVar[Dict[str, List[str]]] = {
        # Radar: vertical-vertical and vertical-horizontal polarizations.
        "S1": ["VV", "VH"],
        # Sentinel-2 Level-1C, all 13 bands (note B8A precedes B8 here —
        # this ordering is what the pretrained SSL4EO backbone expects).
        "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8A", "B8", "B9", "B10", "B11", "B12"],
        # Same as S2L1C but without B10 (cirrus band), 12 bands.
        "S2L1C-12band": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                          "B8A", "B8", "B9", "B11", "B12"],
        # Sentinel-2 Level-2A (surface reflectance); B10 is not present
        # in L2A products at all, hence its absence here too.
        "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8A", "B8", "B9", "B11", "B12"],
        # Google's 64-band annual satellite embedding product: A00..A63.
        "AlphaEarth": [f"A{x:02d}" for x in range(64)],
    }

    # Legacy NDVI band mapping; NDVI-based masking was removed from the
    # active pipeline, but postprocess.py's dead --ndvi_threshold branch
    # still refers to these keys.
    _NDVI_BANDS: ClassVar[Dict[str, str]] = {
        "red": "B4",
        "nir": "B8A"
    }

    def __post_init__(self):
        """Resolve `bands` from `collection` and prepare the image cache dir.

        Called automatically by the dataclass-generated __init__ right
        after fields are set. No inputs beyond `self`; mutates `self` in
        place. Raises ValueError if `collection` has no entry in
        `_BAND_IDS` and `bands` was not explicitly supplied. Side effect:
        creates `image_cache_dir` on disk (with parents) if provided.
        """
        # Automatically resolve bands if not provided
        if self.bands is None:
            if self.collection not in self._BAND_IDS:
                raise ValueError(f"No band mapping defined for collection "
                                 f"'{self.collection}'")
            self.bands = self._BAND_IDS[self.collection]

        if self.image_cache_dir:
            # Normalize to Path and ensure the cache directory exists so
            # later writes (GEE_Data_Extractor.save_tile) do not fail.
            self.image_cache_dir = Path(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def available_collections(cls) -> List[str]:
        """Return the list of supported collection IDs.

        Returns: List[str] of keys of `_BAND_IDS` (e.g. ["S1", "S2L1C",
        "S2L1C-12band", "S2L2A", "AlphaEarth"]). Used by
        inference_pipeline.py to populate argparse's `choices=` for
        --collection. No side effects.
        """
        return list(cls._BAND_IDS.keys())

@dataclass
class InferenceConfig:
    """Configuration for the ML inference stage (embedding + classification).

    Fields:
        model_path: str|Path, filesystem path to the Keras ensemble
            classifier (.h5, architecture+weights, no optimizer state
            needed for inference). Default is a path RELATIVE to the
            gee/ directory ("../models/..."), so scripts using this
            default must be run with gee/ as the current working
            directory.
        pred_threshold: float in [0,1], a chip's mean ensemble confidence
            must be STRICTLY GREATER than this to count as a positive
            detection. Default 0.5 here (0.85 has been used in
            production runs for higher precision).
        embed_model_name: Optional[str], identifier selecting which
            embedding backbone to load; currently only "ssl4eo_vit_s16"
            is supported (see InferenceEngine._load_embed_model).
        embed_model_path: Optional[str], filesystem path to the pickled
            embedding backbone. Default is SSL4EO_PATH. Setting this to
            an empty string disables the embedding backbone entirely —
            the Keras model then receives raw normalized chips directly
            (legacy direct-CNN inference mode, no SSL4EO step).
        embed_model_chip_size: Optional[int], the backbone's expected
            square input size in pixels (SSL4EO ViT-S/16 expects 224).
            Required (non-None) whenever an embedding model is enabled.
        embedding_batch_size: Optional[int], number of chips per forward
            pass through the embedding backbone. Default 32. Required
            whenever an embedding model is enabled.
        geo_chip_size: Optional[int], the chip size in GROUND pixels (at
            the tile's native resolution) cut out of each tile before
            any resizing to the embedding model's input size. Default
            48 px (~480 m per side, ~23 ha). Required whenever an
            embedding model is enabled; if None and no embedding model,
            it is instead inferred from the Keras model's own input
            shape (see InferenceEngine._resolve_chip_params).
        embeddings_cache_dir: Optional[str|Path], if given, per-tile
            embeddings are cached to/read from GeoParquet files in this
            directory, keyed by collection/tile/date range. Default None
            (no caching).
        run_sam2: bool, whether to run SAM2_Masker inline (per positive
            tile) as part of InferenceEngine.bulk_predict. Default False.
        inference_output_base: str|Path, base directory under which a
            per-model-version subfolder of prediction GeoJSONs is
            created. Default DEFAULT_INFERENCE_OUTPUT_BASE
            (<repo>/data/outputs).
        stride_ratio: int, chip stride (in pixels) is computed as
            `chip_size // stride_ratio`. Default 2, so with a 48-px chip
            the stride is 24 px — i.e. chips overlap by 50%, giving
            23x23 = 529 overlapping chips per (padless) 576-px tile.
        tries: int, number of passes bulk_predict will make over tiles
            that failed (I/O or inference errors), retrying only the
            tiles that failed in the previous pass. Default 2.
        max_concurrent_tiles: int, number of tiles processed together in
            one producer/consumer batch inside bulk_predict, bounding
            peak memory use. Default 500.

    Side effects (in __post_init__): normalizes `model_path` and
    `inference_output_base` to Path objects; validates embedding-related
    fields via `_validate_embedding_config`; normalizes
    `embeddings_cache_dir` to a Path and creates it if given; normalizes
    a falsy `embed_model_path` to the empty string "" (the canonical
    "disabled" sentinel).
    """
    # Path to Keras classifier (.h5). Loaded inside :class:`InferenceEngine`.
    model_path: PathLike = (
        "../models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5"
    )
    # A chip counts as a positive detection only if mean confidence > this.
    pred_threshold: float = 0.5
    # Only "ssl4eo_vit_s16" is currently a recognized embedding backbone.
    embed_model_name: Optional[str] = "ssl4eo_vit_s16"
    # Path to the pickled backbone weights; "" disables the backbone.
    embed_model_path: Optional[str] = SSL4EO_PATH
    # The next 3 parameters are required if using an embedding model
    # ViT input resolution (SSL4EO ViT-S/16 expects 224x224 px).
    embed_model_chip_size: Optional[int] = 224
    # Chips per forward pass through the embedding backbone.
    embedding_batch_size: Optional[int] = 32
    # Chip size in ground (native tile) pixels, before any resize to
    # embed_model_chip_size; 48 px ~= 480 m per side ~= 23 ha.
    geo_chip_size: Optional[int] = 48
    # Optional directory for per-tile embeddings parquet cache.
    embeddings_cache_dir: Optional[PathLike] = None
    # Whether to run SAM2 segmentation inline per positive tile.
    run_sam2: bool = False
    # Base directory for prediction GeoJSONs; subfolder per model version at runtime.
    inference_output_base: PathLike = DEFAULT_INFERENCE_OUTPUT_BASE
    stride_ratio: int = 2  # stride is computed as chip_size // stride_ratio.
    # Number of retry passes over tiles that failed to process.
    tries: int = 2
    # Tiles processed per producer/consumer batch (memory bound).
    max_concurrent_tiles: int = 500

    def __post_init__(self):
        """Normalize paths and validate embedding-model configuration.

        Called automatically right after fields are set by the
        dataclass-generated __init__. No inputs beyond `self`; mutates
        `self` in place: coerces `model_path` and
        `inference_output_base` to Path, runs
        `_validate_embedding_config` (may raise ValueError), creates
        `embeddings_cache_dir` on disk if provided, and normalizes a
        falsy `embed_model_path` to "".
        """
        self.model_path = Path(self.model_path)
        self._validate_embedding_config()

        if self.embeddings_cache_dir:
            # Normalize to Path and eagerly create the cache directory.
            self.embeddings_cache_dir = Path(self.embeddings_cache_dir)
            self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.embed_model_path:
            # Canonicalize any falsy value (None, "", etc.) to "" so
            # downstream `if config.embed_model_path:` checks are simple.
            self.embed_model_path = ""

        self.inference_output_base = Path(self.inference_output_base)

    def _validate_embedding_config(self):
        """Ensure required embedding-related fields are set when embedding is enabled.

        No inputs beyond `self`; no return value. Raises ValueError if
        `embed_model_name` and `embed_model_path` are both truthy (i.e.
        an embedding backbone is requested) but any of
        embed_model_chip_size / embedding_batch_size / geo_chip_size is
        None. If embedding is not enabled (either name or path falsy),
        this is a no-op — legacy direct-CNN mode has no such
        requirements.
        """
        if not self.embed_model_name or not self.embed_model_path:
            # Embedding backbone not requested: nothing to validate.
            return

        # Fields that must be set (non-None) for the embedding pipeline
        # (embed() and _resolve_chip_params()) to work correctly.
        required = {
            "embed_model_chip_size": self.embed_model_chip_size,
            "embedding_batch_size": self.embedding_batch_size,
            "geo_chip_size": self.geo_chip_size,
        }

        missing = [k for k,v in required.items() if v is None]
        if missing:
            raise ValueError(
                "Embedding model enabled but missing required parameters: "
                + ", ".join(missing)
            )

@dataclass
class MaskConfig:
    """Configuration for the optional SAM2 segmentation stage.

    Fields:
        prior_sigma: float, pixels, the falloff scale of the quadratic
            spatial prior applied outside classifier-detected chip
            regions (see SAM2_Masker.soft_spatial_prior). Default 12.0 px.
            Larger values let SAM2's mask extend farther from the
            original chip detections before being penalized.
        smoothing_sigma: float, pixels, standard deviation of the
            Gaussian filter applied to the upsampled SAM2 logits before
            thresholding (see SAM2_Masker.upsample_logits). Default 2.5
            px; smooths the final mask boundary.
        sam2_repo_path: str|Path, filesystem path to the cloned SAM2
            repository root (used both to locate Hydra configs and as
            the parent directory for default checkpoint filenames).
            Default SAM2_PATH.
        sam2_checkpoint: Optional[str|Path], path to the base SAM2
            checkpoint weights. Default None, resolved in __post_init__
            to `<sam2_repo_path>/sam2.1_hiera_small.pt`.
        finetuned_weights: Optional[str|Path], path to AMW's fine-tuned
            SAM2 weights (state_dict overlaid on the base checkpoint via
            strict=False). Default None, resolved in __post_init__ to
            `<sam2_repo_path>/SAM_model_96_px_final.pth`.
        sam2_model_cfg: Optional[str|Path], either a Hydra config NAME
            relative to the installed sam2 package (e.g.
            "configs/sam2.1/sam2.1_hiera_s.yaml") or a filesystem path to
            a YAML file located under `<sam2_repo_path>/sam2/configs/`;
            resolved to a package-relative name by
            `_resolve_sam2_hydra_config`. Default None, resolved to
            DEFAULT_SAM2_HYDRA_CONFIG.
        mask_dir: str|Path, output directory for SAM2 mask/logits
            GeoTIFFs. Default DEFAULT_MASK_DIR; created eagerly in
            __post_init__.

    Side effects (in __post_init__): resolves all the above path fields
    (kept as `str`, not `Path`, because the SAM2 library internals expect
    plain string paths); creates `mask_dir` on disk.
    """
    prior_sigma: float = 12.0   # spatial prior sigma (pixels)
    smoothing_sigma: float = 2.5  # gaussian smoothing after upsampling (pixels)

    sam2_repo_path: PathLike = SAM2_PATH
    sam2_checkpoint: Optional[PathLike] = None
    finetuned_weights: Optional[PathLike] = None
    # Hydra config name for ``build_sam2`` (e.g. configs/sam2.1/...), not a path.
    sam2_model_cfg: Optional[PathLike] = None
    mask_dir: PathLike = DEFAULT_MASK_DIR

    def __post_init__(self):
        """Resolve SAM2-related paths to defaults and create mask_dir.

        No inputs beyond `self`; mutates `self` in place. All path
        fields end up as `str` (not `Path`) since the SAM2 library
        expects plain strings. Side effect: creates `mask_dir` (with
        parents) on disk.
        """
        sam2_repo = Path(self.sam2_repo_path)
        self.sam2_repo_path = str(sam2_repo)

        # Keep these as strings: SAM2 internals expect str paths.
        self.sam2_checkpoint = str(
            Path(self.sam2_checkpoint) if self.sam2_checkpoint
            else sam2_repo / "sam2.1_hiera_small.pt"
        )
        self.finetuned_weights = str(
            Path(self.finetuned_weights) if self.finetuned_weights
            else sam2_repo / "SAM_model_96_px_final.pth"
        )
        self.sam2_model_cfg = self._resolve_sam2_hydra_config(
            self.sam2_model_cfg, sam2_repo
        )

        self.mask_dir = Path(self.mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_sam2_hydra_config(
        sam2_model_cfg: Optional[PathLike], sam2_repo: Path
    ) -> str:
        """Return Hydra ``config_name`` for ``sam2.build_sam.build_sam2``.

        Args:
            sam2_model_cfg: either None/empty (use the module default),
                a Hydra package-relative config name already (e.g.
                "configs/sam2.1/sam2.1_hiera_s.yaml"), or a filesystem
                path to a YAML file that lives under
                `<sam2_repo>/sam2/configs/`.
            sam2_repo: Path to the SAM2 repository root, used to locate
                the `sam2/configs/` directory when `sam2_model_cfg` is a
                filesystem path.

        Returns:
            str: a Hydra `config_name` suitable for
            `sam2.build_sam.build_sam2(config_name, ...)` — i.e. a path
            relative to the installed `sam2` package, NOT an absolute
            filesystem path (Hydra's `compose()` requires this; passing
            an absolute path silently fails, see the module-level note
            near DEFAULT_SAM2_HYDRA_CONFIG).

        Side effects: none (pure function; resolves paths on disk only
        to check existence/relativity, does not create anything).
        """
        if not sam2_model_cfg:
            # No override given: fall back to the module-level default.
            return DEFAULT_SAM2_HYDRA_CONFIG
        raw = str(sam2_model_cfg).strip()
        path = Path(raw).expanduser()
        # Directory Hydra configs actually live under, inside the repo.
        configs_root = (sam2_repo / "sam2" / "configs").resolve()
        if path.is_file():
            try:
                # If `raw` is a real file under configs_root, convert it
                # to the package-relative "configs/..." form Hydra needs.
                rel = path.resolve().relative_to(configs_root)
                return f"configs/{rel.as_posix()}"
            except ValueError:
                # File exists but isn't under configs_root; fall through
                # and treat `raw` as an already-relative Hydra name.
                pass
        # Already a Hydra package-relative name (or user override to experiment).
        return raw

class GEE_Data_Extractor:
    """Builds a lazy GEE composite recipe and fetches/caches per-tile pixels.

    This class encapsulates all direct interaction with the Earth Engine
    API: constructing the (server-side, not-yet-executed) image composite
    for a given collection/date-range, and pulling pixel data for
    individual tiles out of that composite via ee.data.computePixels,
    with optional on-disk GeoTIFF caching. It has no knowledge of chips,
    embeddings, or classification — those live in InferenceEngine.

    NOTE (latent bug, not fixed here): the exception handlers in
    get_tile_data reference `self.logger`, but this class's __init__
    never sets a `self.logger` attribute. A cache read/write failure
    inside get_tile_data would therefore itself raise AttributeError
    (masking the original exception) instead of logging a warning and
    continuing.
    """
    def __init__(self, start_date: str, end_date: str, config: DataConfig):
        """Store the date range/config and build the lazy composite recipe.

        Args:
            start_date: str, inclusive start date (YYYY-MM-DD) for the
                GEE ImageCollection filter.
            end_date: str, exclusive/inclusive end date (per GEE
                filterDate semantics) for the collection filter.
            config: DataConfig, collection/band/tiling/cloud-threshold
                settings.

        Side effects: none beyond building the composite recipe, which
        is itself lazy (no network calls / no pixels move) — see
        `_build_composite`.
        """
        # Store dates as plain strings (used later to build cache
        # filenames and output GeoJSON names).
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.config = config
        # Builds the composite ONCE; this is a server-side recipe object
        # (ee.Image), not actual pixel data — nothing is downloaded here.
        self.composite = self._build_composite(start_date, end_date)

    def _build_composite(self, start_date: str, end_date: str):
        """Construct the lazy per-pixel composite ee.Image for `self.config.collection`.

        Args:
            start_date: str, YYYY-MM-DD, passed to ee ImageCollection
                .filterDate.
            end_date: str, YYYY-MM-DD, passed to ee ImageCollection
                .filterDate.

        Returns:
            ee.Image: a server-side composite recipe. For S2 collections
            this is a per-pixel MEDIAN across all scenes in range whose
            individual pixel observations passed the Cloud Score+
            clear-sky threshold; for S1 it is a mosaic (latest scene on
            top) of dual-polarization radar; for AlphaEarth it is a
            mosaic of the annual embedding product. No pixels are moved
            over the network by this call — evaluation is deferred until
            get_tile_data calls ee.data.computePixels.

        Side effects: none (purely builds a lazy ee.Image expression
        graph; all filtering/masking/compositing happens server-side,
        lazily, inside GEE when later evaluated).

        IMPORTANT CAVEAT (S2 branches): nothing here records HOW MANY
        clear (unmasked) observations contributed to each output pixel.
        A pixel that was cloudy in every single scene in the date range
        comes back as fully masked (digital-number 0) with no signal
        that this happened, as opposed to a pixel with a robust median
        of many clear observations. This blind spot — no per-pixel
        "observation count" or cloud-confidence flag survives the
        median() collapse — is exactly what a planned "cloud flag"
        research effort on this fork intends to address.
        """
        collection = self.config.collection
        clear_threshold = self.config.clear_threshold

        if collection in ['S2L1C', 'S2L1C-12band']:
            # Harmonized Sentinel-2 Level-1C (top-of-atmosphere) scenes.
            s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            # Cloud Score+ QA product: per-pixel per-scene "cs_cdf" band
            # is a calibrated probability that the pixel is clear (cloud-
            # and-shadow-free), in [0, 1].
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                # Restrict to scenes overlapping the requested date range.
                s2.filterDate(start_date, end_date)
                # Join each S2 scene to its matching Cloud Score+ image
                # (matched by system:index), attaching the cs_cdf band.
                .linkCollection(csPlus, [QA_BAND])
                # Per-scene, per-pixel masking: any pixel whose cs_cdf is
                # below clear_threshold is masked out of THAT scene only
                # (other scenes' observations of the same ground pixel
                # are unaffected).
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                # Per-pixel median across all surviving (clear) scene
                # observations — a robust composite that tends to reject
                # residual clouds/shadows/other transient artifacts, at
                # the cost of losing information about how many
                # observations (if any) actually contributed.
                .median())

        elif collection == 'S2L2A':
            # Harmonized Sentinel-2 Level-2A (surface reflectance) scenes;
            # same Cloud Score+ masking + median-compositing approach as
            # the L1C branch above, just a different source collection.
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

        elif collection == 'S1':
            # Sentinel-1 Ground Range Detected radar backscatter.
            s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
            composite = (
                s1.filterDate(start_date, end_date)
                # Restrict to Interferometric Wide swath mode (the
                # standard land-imaging acquisition mode).
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                # Require both VV and VH polarizations to be present so
                # the two-band [VV, VH] stack is always fully populated.
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VH"))
                # Radar sees through clouds, so a simple mosaic (latest
                # scene "on top" where scenes overlap) suffices; no
                # cloud-masking/median needed as with optical S2.
                .mosaic())

        elif collection == 'AlphaEarth':
            # Google's annual satellite embedding product (64 learned
            # bands per pixel, not raw reflectance).
            emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            composite = emb.filterDate(start_date, end_date).mosaic()

        else:
            raise ValueError(f'Collection {collection} not recognized.')
        return composite

    @retry.Retry(timeout=240)
    def get_tile_data(self, tile: TileType) -> np.ndarray:
        """Download or load cached image data for a tile.

        Decorated with google-api-core's @retry.Retry(timeout=240): on
        transient GEE errors, this method is automatically retried with
        exponential backoff for up to 240 seconds total before the
        exception is allowed to propagate.

        Inputs:
        - tile: a TileType object
        Outputs:
        - pixels: a numpy array with shape (H, W, bands)

        More precisely:
            tile: TileType (DLTile or CenteredTile). Must expose
                `.key` (unique cache-key string), `.geometry` (shapely
                geometry with `.bounds` in EPSG:4326 lon/lat),
                `.tilesize`, and `.pad` (pixel padding per side).

        Returns:
            np.ndarray, dtype float32, shape (out_size, out_size,
            len(self.config.bands)) where out_size = tile.tilesize +
            2*tile.pad. Values are raw digital numbers (for S2: DN =
            reflectance * 10000, uint16 range before the float32 cast);
            fully cloud-masked pixels come back as 0.

        Side effects:
            - On cache hit (image_cache_dir set and the tile's GeoTIFF
              already exists there): reads from disk via load_tile, no
              network call.
            - On cache miss: issues a synchronous ee.data.computePixels
              call (a real network request to Earth Engine that executes
              the composite recipe over just this tile's footprint), and
              if image_cache_dir is set, writes the result back to disk
              via save_tile.

        LATENT BUG (not fixed here): both the cache-read and cache-write
        except blocks below call `self.logger.warning(...)`, but this
        class never sets `self.logger` in __init__. If a cache read or
        write ever fails, the `except` handler itself raises
        AttributeError (masking the real failure) instead of logging a
        warning and continuing gracefully.
        """
        image_cache_dir = getattr(self.config, "image_cache_dir", None)
        collection = self.config.collection
        start, end = self.start_date, self.end_date
        # Cache key encodes everything that determines the pixel content:
        # collection, tile identity, and date range.
        tif_name = f"{collection}_{tile.key}_{start}_{end}.tif"

        if image_cache_dir:
            tif_path = image_cache_dir / tif_name
            if tif_path.exists():
                try:
                    # Cache hit: skip GEE entirely, load from disk.
                    return self.load_tile(tif_path)
                except Exception as e:
                    # See LATENT BUG note in the docstring above: this
                    # will itself raise AttributeError since self.logger
                    # is never defined on this class.
                    self.logger.warning(
                        f"Failed to load cached tile {tile.key}: {e}")

        # ee.Geometry.Rectangle expects (minx, miny, maxx, maxy) in
        # lon/lat degrees — tile.geometry.bounds is already in that form
        # (tiles are defined/stored in EPSG:4326).
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        # Full output raster size including padding on both sides.
        out_size = tile.tilesize + 2 * tile.pad
        # Server-side clip to the tile footprint AND resample onto an
        # out_size x out_size pixel grid, in one GEE operation. Because
        # the geometry is defined in degrees (EPSG:4326, equirectangular),
        # the resulting pixel spacing is only APPROXIMATELY 10 m — exact
        # ground spacing varies slightly with latitude.
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom,
            width=out_size,
            height=out_size
        )

        # THE key call that actually moves pixels: executes the lazy
        # composite recipe over the clipped/resampled tile footprint and
        # returns real pixel data synchronously (no export-to-Drive/GCS
        # step needed, unlike older GEE workflows) — this API became
        # available in 2023 and is what makes this per-tile pipeline
        # practical.
        pixels = ee.data.computePixels(
            {
                "bandIds": self.config.bands,
                "expression": composite_tile,
                # Structured numpy array: one named field per band.
                "fileFormat": "NUMPY_NDARRAY",
            }
        )

        if isinstance(pixels, np.ndarray):
            if pixels.dtype.fields is not None:
                # NUMPY_NDARRAY comes back as a structured array (each
                # band is a named field, not a stacked axis); restack
                # into an ordinary (H, W, bands) array in the requested
                # band order.
                pixels = np.stack([pixels[band] for band in self.config.bands],
                                 axis=-1)
        else:
            pixels = np.array(pixels)

        # Defensive fix-up for occasional off-by-one raster sizes
        # returned by GEE (pads/crops to exactly out_size x out_size).
        pixels = ensure_tile_shape(pixels, out_size)
        # Cast to float32 for downstream numeric ops (normalization,
        # torch tensors, etc.); copy=False avoids a redundant copy if
        # the array is already float32.
        pixels = pixels.astype(np.float32, copy=False)

        if image_cache_dir:
            try:
                # Cache miss path: persist this tile so future runs with
                # the same collection/tile/date-range can skip GEE.
                self.save_tile(pixels, tile, image_cache_dir)
            except Exception as e:
                # See LATENT BUG note above: self.logger is undefined,
                # so this would itself raise AttributeError.
                self.logger.warning(f"Failed to cache tile {tile.key}: {e}")

        return pixels

    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> List[np.ndarray]:
        """
        Download all tile data concurrently.
        Args:
            tiles: list of tile objects
        Returns:
            - data: list of numpy arrays (one per tile)

        Uses a ThreadPoolExecutor with `self.config.max_workers` threads.
        `executor.map` preserves input order in the returned list (the
        i-th output array corresponds to the i-th input tile), even
        though the underlying downloads complete out of order. Threads
        (not processes) are appropriate here because get_tile_data is
        I/O-bound (network calls to GEE), so the GIL is released during
        the actual wait time.

        Side effects: same as get_tile_data, repeated per tile (network
        calls and/or disk cache reads/writes).
        """
        data = []

        with ThreadPoolExecutor(
            max_workers=self.config.max_workers) as executor:
            # map() blocks until all downloads complete, returning results
            # in the same order as `tiles` was given.
            data = list(executor.map(self.get_tile_data, tiles))

        return data

    @staticmethod
    def affine_from_tile(tile, width: int, height: int):
        """Construct an EPSG:4326 affine transform from tile metadata.

        Args:
            tile: TileType, must expose `.geometry.bounds` (lon/lat
                degrees, EPSG:4326).
            width: int, raster width in pixels.
            height: int, raster height in pixels.

        Returns:
            affine.Affine mapping pixel (col, row) coordinates to
            (lon, lat) geographic coordinates, suitable for use as a
            rasterio dataset's `transform`.

        Side effects: none (pure function).
        """
        return from_bounds(*tile.geometry.bounds, width, height)

    def save_tile(
        self, pixels: np.ndarray, tile: TileType, outdir: Path,
        product_type: Literal["image", "mask", "logits"] = "image") -> Path:
        """
        Write a tile GeoTIFF in EPSG:4326.

        product_type:
            - "image"  -> satellite imagery
            - "mask"   -> uint8 segmentation mask
            - "logits" -> float32 model logits

        Args:
            pixels: np.ndarray, shape (H, W, bands), any numeric dtype
                (will be cast per `product_type` below).
            tile: TileType, source tile (used for its `.key` and
                `.geometry` for georeferencing/filename).
            outdir: Path, destination directory for the GeoTIFF.
            product_type: one of "image" (satellite imagery; dtype
                uint16 for S2-family collections else float32, no
                nodata value, no filename suffix), "mask" (uint8
                segmentation mask; nodata=2; filename suffix "-msk"), or
                "logits" (float32 raw model logits; nodata=NaN; filename
                suffix "-logits").

        Returns:
            Path to the written GeoTIFF file
            (`outdir/<collection>_<tile.key>_<start>_<end><suffix>.tif`).

        Side effects: writes a compressed, internally tiled GeoTIFF file
        to disk at the returned path (overwriting any existing file with
        that name).
        """
        if product_type == "mask":
            # Binary segmentation mask: 0/1 values, small dtype; nodata=2
            # is used as a sentinel distinct from the valid {0, 1} values.
            dtype = "uint8"
            nodata = 2
            suffix = "-msk"
        elif product_type == "logits":
            # Raw (possibly negative) model logits need float precision.
            dtype = "float32"
            nodata = np.nan
            suffix = "-logits"
        else:
            # "image": satellite imagery. S2 collections store raw
            # digital numbers as uint16 (DN = reflectance * 10000); other
            # collections (S1 radar, AlphaEarth embeddings) use float32
            # since their values are not simple scaled reflectance.
            if self.config.collection[:2] == "S2":
                dtype = "uint16"
            else:
                dtype = "float32"
            nodata = None
            suffix = ""

        # rasterio/GDAL expect band-major arrays: (bands, H, W), so move
        # the last (band) axis of the (H, W, bands) input to the front.
        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        transform = self.affine_from_tile(tile, width, height)
        crs = "EPSG:4326"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            # deflate compression + internal tiling: smaller files and
            # faster partial/windowed reads for large tiles.
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

        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(pixels)

        return outpath


    def load_tile(self, path: Path) -> np.ndarray:
        """Load a tile’s pixels from GeoTIFF.

        Args:
            path: Path to a GeoTIFF previously written by `save_tile`
                (or otherwise readable by rasterio).

        Returns:
            np.ndarray, dtype float32, shape (H, W, bands) — i.e. the
            inverse of the band-major (bands, H, W) layout rasterio
            reads natively.

        Side effects: opens and reads the file at `path` (read-only).
        """
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
            arr = np.moveaxis(arr, 0, -1)  # back to (H,W,B)
        return arr

class InferenceEngine:
    """Handles tile inference, retries, and aggregation."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        data_config: DataConfig,
        config: InferenceConfig,
        mask_config: Optional[MaskConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Build the data extractor, load the classifier (and optional backbone/masker).

        Args:
            start_date: str, YYYY-MM-DD, passed through to
                GEE_Data_Extractor for the composite date filter.
            end_date: str, YYYY-MM-DD, passed through to
                GEE_Data_Extractor.
            data_config: DataConfig, GEE acquisition/tiling settings.
            config: InferenceConfig, model paths, embedding/classifier
                settings, concurrency settings.
            mask_config: Optional[MaskConfig], SAM2 settings; only used
                (and defaulted if None) when `config.run_sam2` is True.
            logger: Optional[logging.Logger]; defaults to the root
                logger via `logging.getLogger()` if not supplied.

        Raises:
            ValueError: if `config.model_path` is falsy (the Keras
            classifier path is mandatory).

        Side effects (all happen synchronously during construction, i.e.
        this constructor is not cheap):
            - Builds the GEE composite recipe (GEE_Data_Extractor.__init__;
              lazy, no network call yet).
            - Loads the Keras ensemble classifier from disk
              (tf.keras.models.load_model), a real disk read + model
              deserialization.
            - Creates a threading.Lock to serialize Keras predict() calls
              (Keras models are not thread-safe for concurrent inference).
            - Selects a torch device: cuda > mps (Apple GPU) > cpu.
            - If `config.embed_model_path` is set, loads the SSL4EO
              PyTorch backbone from disk (torch.load, unpickles a full
              nn.Module) and moves it to the selected device.
            - If `config.run_sam2` is True, constructs a SAM2_Masker,
              which itself loads SAM2 model weights from disk.
        """
        self.data_extractor = GEE_Data_Extractor(
            start_date, end_date, data_config)
        self.config = config
        self.logger = logger or logging.getLogger()

        if not config.model_path:
            raise ValueError("InferenceConfig.model_path must be set to a Keras .h5 file")
        # Load the Keras ensemble classifier. compile=False: no optimizer
        # state is needed for inference-only use, and the architecture is
        # fully self-described inside the HDF5 file's JSON config, so no
        # custom layer code needs to be registered here.
        self.model = tf.keras.models.load_model(
            str(config.model_path), compile=False)

        # Add a lock to serialize model access (in-process)
        # Keras's model.predict is not guaranteed thread-safe; this lock
        # ensures only one thread calls into the model at a time even
        # though many producer threads may be feeding work concurrently.
        self._tf_model_lock = threading.Lock()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Apple Silicon GPU backend (Metal Performance Shaders).
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if config.embed_model_path:
            # Embedding backbone enabled: load it now (once) rather than
            # per-tile.
            self.embed_model = self._load_embed_model()
        else:
            # Legacy mode: Keras model will receive raw chips directly.
            self.embed_model = None

        if self.config.run_sam2:
            if not mask_config:
                # No explicit MaskConfig supplied but SAM2 requested:
                # fall back to library defaults.
                mask_config = MaskConfig()
            self.masker = SAM2_Masker(self.data_extractor, mask_config)
        else:
            self.masker = None

    def predictions_geojson_path(self, region_name: str) -> Path:
        """Path for bulk-inference positive-chip GeoJSON under ``inference_output_base``.

        ``region_name`` is a label for the output file only (e.g. ``Path(geojson).stem``);
        it is not used to load geometry.

        Uses ``config.model_path`` stem (first two ``_``-separated tokens as model
        version), ``config.pred_threshold``, ``config.inference_output_base``, and
        the data extractor's date range.

        Args:
            region_name: str, arbitrary label included in the output
                filename (typically the region GeoJSON's file stem);
                purely cosmetic, does not affect which geometry is used.

        Returns:
            Path: `<inference_output_base>/<model_version>/
            <region_name>_<model_version>_<pred_threshold:.2f>_
            <start_date>_<end_date>.geojson`, where `model_version` is
            the first two underscore-separated tokens of the model
            file's stem (e.g. a model file named
            "48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5" yields
            model_version "48px_v0.X-SSL4EO-MLPensemble").

        Raises:
            ValueError: if `self.config.model_path` is falsy.

        Side effects: creates the per-model-version output directory on
        disk (mkdir with parents) if it does not already exist.
        """
        if not self.config.model_path:
            raise ValueError(
                "InferenceConfig.model_path must be set to build output path"
            )
        mp = Path(self.config.model_path)
        # e.g. "48px_v0.X-SSL4EO-MLPensemble_2025-10-21" ->
        # "48px_v0.X-SSL4EO-MLPensemble" (drop the trailing date token).
        model_version = "_".join(mp.stem.split("_")[:2])
        period = f"{self.data_extractor.start_date}_{self.data_extractor.end_date}"
        outdir = self.config.inference_output_base / model_version
        outdir.mkdir(parents=True, exist_ok=True)
        pred = self.config.pred_threshold
        return outdir / (
            f"{region_name}_{model_version}_{pred:.2f}_{period}.geojson"
        )

    def _load_embed_model(self):
        """Load and prepare the configured embedding backbone.

        No inputs beyond `self` (reads `self.config.embed_model_name`
        and `self.config.embed_model_path`).

        Returns:
            torch.nn.Module: the loaded backbone, moved to `self.device`
            and set to eval() mode (disables dropout/batchnorm training
            behavior, relevant for correct, deterministic inference).

        Raises:
            ValueError: if `self.config.embed_model_name` is not a
            recognized identifier (currently only "ssl4eo_vit_s16" is
            supported).

        Side effects: reads the backbone weights file from disk via
        torch.load (see security note below); moves the resulting module
        to `self.device` (GPU/MPS memory allocation if applicable).

        SECURITY NOTE: `torch.load(..., weights_only=False)` unpickles a
        FULL nn.Module (architecture + weights in a single pickle blob),
        not just a state_dict. Unlike weights_only=True, this can execute
        arbitrary code during unpickling — only ever point
        embed_model_path at a trusted local file.
        """
        if self.config.embed_model_name == "ssl4eo_vit_s16":
            # weights_only=False: this is a full pickled nn.Module (see
            # SECURITY NOTE above), not merely a state_dict.
            embed_model = torch.load(self.config.embed_model_path,
                                     weights_only=False)
        else:
            raise ValueError(
                f"Unknown embedding model: {self.config.embed_model_name}")

        embed_model = embed_model.to(self.device)
        # eval() mode: disables dropout and freezes batchnorm running
        # stats, appropriate for a frozen feature-extractor backbone.
        embed_model.eval()
        return embed_model

    def _make_embedding_cache_path(self, tile: TileType) -> Optional[Path]:
        """Return Path to an embedding cache file; None if disabled.

        Args:
            tile: TileType, must expose `.key` (unique cache-key string).

        Returns:
            Optional[Path]: `<embeddings_cache_dir>/<collection>_
            <tile.key>_<start>_<end>_embeddings.parquet`, or None if
            `self.config.embeddings_cache_dir` is not set (caching
            disabled).

        Side effects: none (pure path construction; does not touch disk).
        """
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is None:
            return None

        collection = self.data_extractor.config.collection
        start = self.data_extractor.start_date
        end = self.data_extractor.end_date
        emb_name = f"{collection}_{tile.key}_{start}_{end}_embeddings.parquet"
        return emb_cache_dir / emb_name

    def embed(
        self,
        chips: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: Optional[TileType] = None) -> np.ndarray:
        """Embed chips via a foundation model, with optional caching.

        Args:
            chips: np.ndarray, dtype float32, shape (N, geo_chip_size,
                geo_chip_size, bands) — N normalized (already scaled to
                roughly [0,1] reflectance) image chips in NHWC layout,
                as produced by tile_utils.cut_chips.
            chip_geoms: gpd.GeoDataFrame with one row per chip (same
                order/length as `chips`), a "geometry" column of chip
                footprint polygons, used only for embedding-cache
                writes (georeferenced embeddings) — not needed for the
                forward pass itself.
            tile: Optional[TileType]; if provided, used to compute an
                embedding cache path (read-then-write-through cache);
                if None, caching is skipped entirely for this call.

        Returns:
            np.ndarray, dtype float32, shape (N, 384) — one 384-d SSL4EO
            ViT-S/16 embedding vector per input chip, same row order as
            `chips`/`chip_geoms`.

        Side effects:
            - Cache hit (tile given, cache file exists, loads OK): reads
              the parquet file from disk, no GPU/CPU forward pass run.
            - Cache miss: runs the embedding backbone forward pass
              (uses `self.device`, e.g. GPU); if `tile` is given, writes
              a new GeoParquet cache file with a geometry column and one
              float column per embedding dimension (named f0..f383).
        """

        emb_path = (
            self._make_embedding_cache_path(tile) if
            tile is not None else None
        )

        if emb_path is not None and emb_path.exists():
            try:
                # Cache hit: load previously computed embeddings and skip
                # the (comparatively expensive) backbone forward pass.
                gdf = gpd.read_parquet(emb_path)
                embeddings = gdf.drop(
                    columns="geometry",
                    errors='ignore').to_numpy(dtype=np.float32)
                return embeddings
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cached embeddings for {tile.key}: {e}")

        model_chip_size = self.config.embed_model_chip_size
        geo_chip_size = self.config.geo_chip_size

        # NHWC (numpy/TF convention) -> NCHW (PyTorch convention): permute
        # axes (N, H, W, C) -> (N, C, H, W).
        tensor = torch.from_numpy(chips).permute(0, 3, 1, 2)  # NHWC → NCHW
        if geo_chip_size != model_chip_size:
            # Resize ground chips (e.g. 48x48 px) up to the backbone's
            # expected input resolution (e.g. 224x224 px) using bicubic
            # interpolation; align_corners=False is the standard/default
            # convention matching how the backbone was likely trained.
            tensor = F.interpolate(
                tensor, size=(model_chip_size, model_chip_size),
                mode='bicubic', align_corners=False)

        embeddings_list = []
        batch_size = self.config.embedding_batch_size
        # torch.no_grad(): disables autograd graph construction, saving
        # memory and time since we never backpropagate through a frozen
        # inference-only backbone.
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                # Move just this batch to the compute device and ensure
                # float32 dtype (matches backbone's expected input dtype).
                batch = tensor[i:i+batch_size].to(
                    self.device, dtype=torch.float32)
                out = self.embed_model(batch)
                if isinstance(out, dict):
                    # Some backbone wrappers return a dict of named
                    # outputs; take the first (assumed primary) tensor.
                    out = out[list(out.keys())[0]]
                # Move each batch's output back to CPU immediately so GPU
                # memory doesn't accumulate across all batches.
                embeddings_list.append(out.cpu())

        # Concatenate all batches along the sample (N) axis, then convert
        # to a plain numpy array: shape (N, 384).
        embeddings = torch.cat(embeddings_list, dim=0).numpy()

        if emb_path is not None:
            try:
                # Persist embeddings as a georeferenced GeoParquet: each
                # embedding is tied to its chip's footprint geometry, so
                # the cache is independently reusable (e.g. for training
                # new classifier heads without re-downloading imagery or
                # re-running the backbone).
                gdf = gpd.GeoDataFrame(
                    embeddings,
                    columns=[f"f{i}" for i in range(embeddings.shape[1])],
                    geometry=chip_geoms["geometry"],
                    crs="EPSG:4326"
                )
                gdf.to_parquet(emb_path, index=False)
            except Exception as e:
                self.logger.warning(
                    f"Failed to save embeddings cache for {tile.key}: {e}")

        return embeddings

    def _resolve_chip_params(self):
        """Return (chip_size, stride) based on model/config.

        No inputs beyond `self`. If `self.config.geo_chip_size` is set,
        it is used directly as the chip size. Otherwise (legacy
        direct-CNN mode with no explicit geo_chip_size), the chip size
        is INFERRED from the Keras model's first layer's input shape —
        handling both a plain Sequential/Functional model (input_shape
        is a single tuple) and an ensemble wrapper (input_shape is a
        list of tuples, one per sub-model; the first is used).

        Returns:
            Tuple[int, int]: (chip_size, stride) both in pixels, where
            `stride = chip_size // self.config.stride_ratio`. With the
            default stride_ratio of 2, chips overlap by 50%.

        Side effects: none (read-only; may access `self.model.layers`).
        """
        if self.config.geo_chip_size is None:
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):  # ensemble of models
                # For an ensemble, input_shape is a list of per-submodel
                # shapes (e.g. [(None, 48, 48, 13), (None, 48, 48, 13),
                # ...]); all members are expected to share the same
                # spatial chip size, so just take the first.
                input_shape = input_shape[0]
            # input_shape is (batch, H, W, C); index 1 is the chip's
            # spatial height/width (assumed square).
            chip_size = input_shape[1]
        else:
            chip_size = self.config.geo_chip_size
        stride = chip_size // self.config.stride_ratio
        return chip_size, stride

    def _model_infer(self, x):
        """Thread-safe wrapper around model.predict for inference.

        Args:
            x: np.ndarray, either shape (N, 384) — embeddings, when an
                embedding backbone is in use — or shape (N, chip_size,
                chip_size, bands) — raw chips, in legacy direct-CNN mode
                with no embedding backbone.

        Returns:
            np.ndarray of model outputs. For the current 6-head sigmoid
            ensemble architecture, shape (N, 6); other architectures
            (single sigmoid, softmax binary) return (N, 1) or (N, 2)
            respectively (see `_preds_to_gdf` for how each shape is
            interpreted).

        Side effects: acquires `self._tf_model_lock` for the duration of
        the call, serializing concurrent calls into the (not
        thread-safe) Keras model across producer/consumer threads.
        """
        with self._tf_model_lock:
            return self.model.predict(x, verbose=0)

    def _preds_to_gdf(self, preds: np.ndarray,
                      chip_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert model preds -> preds_gdf

        Args:
            preds: np.ndarray of raw model outputs from `_model_infer`;
                interpreted based on its shape (see branches below).
            chip_geoms: gpd.GeoDataFrame, one row per chip (same order
                and length as `preds`' first axis), with a "geometry"
                column of chip footprint polygons.

        Returns:
            gpd.GeoDataFrame, CRS EPSG:4326, containing only the chips
            whose mean confidence strictly exceeds
            `self.config.pred_threshold`, with columns:
              - "geometry": the chip's footprint polygon.
              - "confidence": float, the (per-chip) mean prediction used
                for thresholding.
              - "preds" (only present when preds is an M>2-head sigmoid
                ensemble): str(list(...)) of that chip's individual
                per-head scores — stringified because GeoJSON feature
                properties must be scalar values, not lists; this
                doubles as a free per-chip uncertainty signal (how much
                the ensemble members agree). Returns an EMPTY
                GeoDataFrame (columns=["geometry", "confidence"],
                crs=EPSG:4326) if no chip passes the threshold.

        Side effects: none (pure transform).
        """
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                # sigmoid binary classifier
                mean_preds = preds.squeeze()
            elif preds.shape[1] == 2:
                # softmax binary classifier
                mean_preds = preds[:, 1]
            else:
                # ensemble of M>2 sigmoid models
                mean_preds = np.mean(preds, axis=1)
                # (Note: M=2 ensemble would be misconstrued as softmax binary)
        else:
            # already shape (N,)
            mean_preds = preds

        # STRICT inequality: a chip exactly AT pred_threshold does not count.
        idx = np.where(mean_preds > self.config.pred_threshold)[0]
        if len(idx) == 0:
            # No positives: return an empty, but correctly-schema'd, gdf
            # so downstream concatenation/CRS logic doesn't need special
            # casing for "no detections this tile".
            return gpd.GeoDataFrame(
                columns=["geometry", "confidence"], crs="EPSG:4326")

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"].reset_index(drop=True),
            crs="EPSG:4326"
        )
        preds_gdf["confidence"] = mean_preds[idx]
        if preds.ndim == 2 and preds.shape[1] > 2:
            # Ensemble case: also keep each member's individual score per
            # chip, stringified (GeoJSON properties must be scalars) as a
            # per-chip uncertainty/agreement signal.
            preds_gdf["preds"] = [str(list(v)) for v in preds[idx]]

        preds_gdf = preds_gdf.set_crs("epsg:4326", allow_override=True)
        return preds_gdf

    def _ensure_gdf(self, df):
        """Ensure input is a GeoDataFrame with geometry column and CRS.

        Returns: GeoDataFrame
        Raises: ValueError: If df is not a GeoDataFrame and not empty.

        Args:
            df: a pandas/geopandas object, typically the result of
                `pd.concat([...])` over a list of GeoDataFrames — which
                can silently degrade to a plain (non-Geo) DataFrame when
                the list is empty or all-empty, losing geometry/CRS
                metadata.

        Side effects: none (pure; returns a new object or the input
        unchanged, never mutates `df` in place).
        """
        if isinstance(df, gpd.GeoDataFrame):
            # Already fine (concat preserved GeoDataFrame-ness).
            return df
        elif df.empty:
            # pd.concat of GeoDataFrames degraded to a plain empty
            # DataFrame: re-wrap it with the expected geometry/CRS.
            return gpd.GeoDataFrame(df, geometry="geometry", crs='epsg:4326')
        else:
            # Non-empty but not a GeoDataFrame: something unexpected
            # happened upstream; fail loudly rather than silently drop
            # geometry information.
            raise ValueError(f"Expected a gdf, got {type(df)} length {len(df)}")

    def produce_tile_input(self, tile: TileType) -> Dict[str, Any]:
        """
        Producer work for a single tile:
        - try to load cached embeddings (if embeddings_cache_dir configured)
        - if cached embeddings exist and load OK -> return {'mode':'embeddings',
            'embeddings':..., 'chip_geoms':..., 'tile': tile}
        - otherwise fetch pixels via get_tile_data -> return {'mode':'pixels',
            'pixels':..., 'tile': tile}
        Errors are raised to the caller.

        Note: When mode is ``embeddings``, the consumer runs classification only
        from cached vectors (no tile pixels in the queue). Inline SAM2 masking
        (``InferenceConfig.run_sam2``) is therefore skipped for those tiles even
        if there are positive detections. For production runs with SAM2, leave
        ``embeddings_cache_dir`` unset or avoid hitting the cache for tiles that
        need masks; use the standalone SAM2 scripts for a separate masking pass.

        This method is designed to run inside a ThreadPoolExecutor as the
        "producer" side of InferenceEngine.bulk_predict's producer/
        consumer pipeline: many of these can run concurrently (I/O
        bound), feeding a single serialized consumer thread that does
        the GPU/TF work.

        Args:
            tile: TileType, the tile to fetch input for.

        Returns:
            Dict[str, Any]: either
              {"mode": "embeddings", "embeddings": np.ndarray (N,384),
               "chip_geoms": gpd.GeoDataFrame, "tile": tile}
            or
              {"mode": "pixels", "pixels": np.ndarray (H,W,bands),
               "tile": tile}

        Raises:
            Any exception raised by `get_tile_data` (e.g. after
            exhausting its internal retry budget) propagates unhandled
            to the caller (bulk_predict), which records the tile as
            failed for a later retry pass.

        Side effects: possible cache file read (embeddings) and/or a
        real GEE network call + optional cache write (pixels, via
        get_tile_data).
        """
        # Try loading cached embeddings first (best-case fast path).
        emb_path = self._make_embedding_cache_path(tile)
        if emb_path and emb_path.exists():
            try:
                embeddings_gdf = gpd.read_parquet(emb_path)
                # Keep only the geometry column for chip footprints...
                chip_geoms = embeddings_gdf[["geometry"]].copy()
                # ...and everything else (f0..f383) becomes the raw
                # embedding matrix.
                embeddings = embeddings_gdf.drop(
                    columns="geometry",
                    errors="ignore").to_numpy(dtype=np.float32)
                return {
                    "mode": "embeddings",
                    "embeddings": embeddings,
                    "chip_geoms": chip_geoms,
                    "tile": tile
                }
            except Exception as e:
                self.logger.warning(
                    f"Failed to load embedding for {tile.key}: {e}. "
                    f"Will fetch pixels.")

        # If no usable embeddings cache, fetch pixels (I/O bound).
        pixels = self.data_extractor.get_tile_data(tile)
        return {"mode": "pixels", "pixels": pixels, "tile": tile}

    def predict_on_tile_pixels(
        self,
        pixels: np.ndarray,
        tile: TileType) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Run the per-tile pipeline starting from pixels (I/O already done).
        This is intended to run inside the consumer (serialized for GPU use).

        Args:
            pixels: np.ndarray, shape (H, W, bands), raw digital-number
                pixel values as returned by GEE_Data_Extractor.get_tile_data
                (for S2, DN = reflectance * 10000).
            tile: TileType, the source tile (used for its `.geometry`,
                `.tilesize`, `.pad`, and `.key`).

        Returns:
            Tuple[gpd.GeoDataFrame, Optional[TileType]]: (preds_gdf,
            failed_tile). On success, `failed_tile` is None and
            `preds_gdf` holds the positive-chip detections (possibly
            empty) for this tile. On failure, `preds_gdf` is an empty
            GeoDataFrame and `failed_tile` is `tile` itself, signaling
            the caller (bulk_predict/consumer) to retry this tile in a
            later pass.

        Side effects:
            - If an embedding backbone is configured: calls `self.embed`
              (possible cache read/write + GPU forward pass).
            - Always: calls `self._model_infer` (Keras predict, GPU/CPU,
              serialized via the TF lock).
            - If there are positive detections AND `self.masker` is set
              (SAM2 enabled): runs inline SAM2 segmentation, which
              itself writes mask/logits GeoTIFFs to disk.
            - Logs a warning if the tile's pixel width is not evenly
              divisible by the chip stride (meaning the chip grid will
              not perfectly tile the full tile, leaving some edge pixels
              unclassified).
        """
        chip_size, stride = self._resolve_chip_params()
        tile_width = tile.tilesize + 2 * tile.pad
        if tile_width % stride != 0:
            # If the tile width doesn't divide evenly by stride, the
            # sliding chip window will not exactly reach the tile's far
            # edge, silently leaving a strip of pixels unclassified.
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={self.config.stride_ratio}). "
                f"Inference may miss some pixels."
            )

        # Convert raw digital numbers to approximate surface reflectance
        # in [0, 1]: divide by 10000 (S2 DN scaling), then clip to guard
        # against occasional values above 1.0 (e.g. from clouds/specular
        # highlights/sensor artifacts) that would otherwise leak out-of-
        # range values into the embedding backbone / classifier.
        normed = np.clip(pixels / 10000.0, 0, 1).astype(np.float32)
        # Slice the (H, W, bands) tile into overlapping square chips:
        # with default 576px tile, 48px chip, 24px stride, this yields
        # 23 x 23 = 529 chips, shape (529, 48, 48, bands), plus a
        # parallel GeoDataFrame of their EPSG:4326 footprint geometries.
        chips, chip_geoms = cut_chips(
            normed, tile.geometry.bounds, chip_size, stride, crs='epsg:4326')

        try:
            if self.embed_model is not None:
                # Modern path: embed chips with the frozen backbone, then
                # classify the resulting 384-d vectors.
                embeddings = self.embed(chips, chip_geoms, tile)
                preds = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                preds_gdf, failed_tile = preds
                if failed_tile is not None:
                    # Classification itself failed: propagate failure
                    # immediately (empty gdf + failed tile) rather than
                    # attempting masking on garbage results.
                    return gpd.GeoDataFrame(), failed_tile
            else:
                # Legacy path: no embedding backbone, classify raw chips
                # directly with the Keras model (expects (N, chip, chip,
                # bands) input).
                preds = self._model_infer(chips)
                preds_gdf = self._preds_to_gdf(preds, chip_geoms)

            if not preds_gdf.empty and self.masker:
                # Inline SAM2 segmentation: only runs when this tile had
                # at least one positive chip AND SAM2 masking is enabled.
                self.masker.predict(pixels, tile, preds_gdf)

            return preds_gdf, None

        except Exception as e:
            # Any failure anywhere in the try block (embedding, model
            # inference, or SAM2 masking) results in this tile being
            # marked failed for retry, rather than a partial/corrupt
            # result being returned.
            self.logger.error(f"Error predicting for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile

    def predict_on_tile_embeddings(
        self,
        embeddings: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: TileType) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """Run the TF classifier on already-available embeddings.

        Args:
            embeddings: np.ndarray, dtype float32, shape (N, 384) —
                already-computed per-chip embedding vectors (from cache
                or from `embed()`).
            chip_geoms: gpd.GeoDataFrame, one row per chip (same order
                as `embeddings`), with chip footprint "geometry".
            tile: TileType, used only for error logging (its `.key`).

        Returns:
            Tuple[gpd.GeoDataFrame, Optional[TileType]]: (preds_gdf,
            failed_tile), following the same (result, failed-tile-or-
            None) convention as `predict_on_tile_pixels`.

        Side effects: calls `self._model_infer` (Keras predict, GPU/CPU,
        serialized via the TF lock); logs an error on failure.
        """
        try:
            preds = self._model_infer(embeddings)
            preds_gdf = self._preds_to_gdf(preds, chip_geoms)
            return preds_gdf, None
        except Exception as e:
            tile_key = getattr(tile, "key", "unknown")
            self.logger.error(f"Error predicting on embeddings for tile "
                              f"{tile_key}: {e}")
            return gpd.GeoDataFrame(), tile

    def _consumer(
        self, q: "queue.Queue[Dict[str, Any]]", sentinel: object,
        nonlocal_predictions: List[gpd.GeoDataFrame],
        fails: List[TileType], consumer_done: threading.Event,
        lock: threading.Lock):
        """Consume items from the queue, serially running inference.

        This runs as a single dedicated (daemon) thread inside
        bulk_predict, forming the "consumer" half of the producer/
        consumer pipeline: it processes exactly one queue item at a time
        (via the blocking `q.get()` loop below), which is what keeps all
        GPU/Keras/PyTorch inference calls serialized even though many
        producer threads may be fetching tile inputs concurrently.

        Args:
            q: queue.Queue[Dict[str, Any]], items produced by
                `produce_tile_input` (mode "embeddings" or "pixels"), or
                the unique `sentinel` object signaling no more work.
            sentinel: object, a unique per-batch sentinel value; compared
                by identity (`is`) so it cannot be confused with a
                genuine (dict) work item.
            nonlocal_predictions: List[gpd.GeoDataFrame], mutated
                in-place: successful non-empty per-tile prediction
                GeoDataFrames are appended here (this is how results
                escape the consumer thread back to the caller).
            fails: List[TileType], mutated in-place: tiles whose
                processing failed are appended here for a later retry
                pass.
            consumer_done: threading.Event, SET in the `finally` block
                once this method returns (success or failure) — the main
                thread in bulk_predict blocks on `.wait()` for this event
                so it never proceeds (or deadlocks) before the consumer
                has actually finished draining the queue.
            lock: threading.Lock, guards concurrent-safe appends to
                `nonlocal_predictions`/`fails` (defensive; in the current
                design there is only one consumer thread, but this keeps
                the shared-state mutation explicitly synchronized).

        Returns: None (all results are communicated via the mutable
        `nonlocal_predictions`/`fails` lists and the `consumer_done`
        event, not a return value — this method runs in its own thread).

        Side effects: blocks on `q.get()` until items (or the sentinel)
        arrive; calls into `predict_on_tile_embeddings` /
        `predict_on_tile_pixels` (GPU/CPU inference, possible SAM2 I/O);
        calls `q.task_done()` after each processed item; always sets
        `consumer_done` before returning, even on an unhandled exception,
        so the caller's wait() never hangs forever.
        """
        try:
            while True:
                # Blocks until a producer (or the final sentinel put)
                # makes an item available.
                item = q.get()
                if item is sentinel:
                    # Identity check (not equality): guarantees only the
                    # exact sentinel object terminates the loop.
                    break

                mode = item.get("mode")
                if mode == "embeddings":
                    embeddings = item["embeddings"]
                    chip_geoms = item["chip_geoms"]
                    tile_local = item["tile"]
                    preds = self.predict_on_tile_embeddings(
                        embeddings, chip_geoms, tile_local)
                elif mode == "pixels":
                    pixels = item["pixels"]
                    tile_local = item["tile"]
                    preds = self.predict_on_tile_pixels(pixels, tile_local)
                else:
                    # Defensive branch: should not happen given
                    # produce_tile_input's two possible modes, but avoid
                    # crashing the whole consumer thread on unexpected
                    # input.
                    self.logger.error(f"Unknown queue mode: {mode}")
                    preds = (gpd.GeoDataFrame(), item.get("tile"))
                preds_gdf, failed_tile = preds

                with lock:
                    if failed_tile is not None:
                        fails.append(failed_tile)
                    else:
                        if not preds_gdf.empty:
                            nonlocal_predictions.append(preds_gdf)

                q.task_done()

        except Exception as e:
            # Catch-all so an unexpected error doesn't leave the
            # consumer thread silently dead without ever setting
            # consumer_done (which would hang the caller's .wait()).
            self.logger.error(f"Consumer error: {e}")
        finally:
            consumer_done.set()

    def bulk_predict(
        self,
        tiles: List[TileType],
        region_name: str,
    ) -> gpd.GeoDataFrame:
        """
        Producer-consumer bulk inference, with retry logic:
         - producers attempt to load embeddings cache; failing, fetch pixels
         - consumer serializes GPU work: embedding model (if required) and
          TF classifier

        If ``embeddings_cache_dir`` is set and a tile is served from cache
        (``mode == "embeddings"``), inline SAM2 masking does not run for that
        tile (see ``produce_tile_input``). Typical production inference without
        an embeddings cache is unaffected.

        ``region_name`` labels the output GeoJSON only (e.g. ``region_path.stem``).

        Writes cumulative predictions to the GeoJSON path from
        :meth:`predictions_geojson_path` (after each batch merge).

        THE main orchestration entry point for a bulk inference run. High
        level algorithm:
          1. Build the output path and an empty cumulative predictions
             GeoDataFrame.
          2. Loop up to `self.config.tries` times, each pass over
             whatever tiles failed in the previous pass (`retry_tiles`,
             initially all of `tiles`).
          3. Within a pass, process `retry_tiles` in batches of
             `self.config.max_concurrent_tiles` tiles, to bound peak
             memory (each tile's raw pixel array is sizeable, e.g. a
             576x576x13 float32 tile is ~17 MB).
          4. Per batch: spin up a single consumer thread reading from a
             bounded `queue.Queue(maxsize=64)` (this bound provides
             BACKPRESSURE — producer threads block on `q.put` once 64
             items are queued, capping how far ahead of the consumer the
             producers can race, and thus capping peak RAM used by
             in-flight tile pixel arrays); submit all of this batch's
             tiles to a ThreadPoolExecutor running `produce_tile_input`;
             as each producer future completes, push its result onto the
             queue (or record a failure); once all producers are done,
             push the sentinel and wait for the consumer to drain and
             finish.
          5. Merge this batch's collected predictions into the
             cumulative GeoDataFrame and REWRITE THE FULL cumulative
             GeoJSON to disk — i.e. every batch fully overwrites the
             output file with the running total so far, which acts as a
             checkpoint: a crash mid-run loses at most the tiles from the
             batch currently in progress.
          6. After a pass, `retry_tiles` becomes the list of tiles that
             failed during that pass, and `tries_remaining` decrements;
             the outer loop exits once either `tries_remaining` hits 0 or
             there are no more failed tiles to retry.

        Args:
            tiles: List[TileType], all tiles to run inference over.
            region_name: str, label used to build the output GeoJSON
                filename (see `predictions_geojson_path`).

        Returns:
            gpd.GeoDataFrame, CRS EPSG:4326, the cumulative set of
            positive-chip detections across all tiles and all retry
            passes (same object that was progressively written to
            `outpath`).

        Side effects:
            - Many concurrent GEE network calls (via `produce_tile_input`
              -> `get_tile_data`).
            - Many Keras/PyTorch inference calls (serialized in the
              consumer thread).
            - Possible inline SAM2 mask/logits GeoTIFF writes.
            - Repeatedly overwrites the predictions GeoJSON at `outpath`
              (once per batch, across all passes) — the LAST write is the
              final, complete result.
            - Logs progress (tries remaining, new positives per batch,
              failed-tile counts) and prints a "Found N new positives"
              line to stdout per batch.
        """
        outpath = self.predictions_geojson_path(region_name)

        # Cumulative results accumulator, pre-seeded with the correct
        # schema/CRS so it can be concatenated with batch results even
        # before any predictions exist.
        predictions = gpd.GeoDataFrame({
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "confidence": gpd.pd.Series(dtype="float"),
            },
            crs="epsg:4326"
        )

        # Start with every tile "to be tried"; shrinks to just the
        # failures after each pass.
        retry_tiles = tiles.copy()
        tries_remaining = self.config.tries
        max_concurrent_tiles = self.config.max_concurrent_tiles
        max_workers = self.data_extractor.config.max_workers

        while tries_remaining and retry_tiles:
            self.logger.info(f"{tries_remaining} tries remaining.")
            fails: List[TileType] = []

            # Process tiles in fixed-size batches to bound the number of
            # in-flight raw pixel arrays (memory) at any one time.
            for i in tqdm(range(0, len(retry_tiles), max_concurrent_tiles)):
                batch_tiles = retry_tiles[i : i + max_concurrent_tiles]
                # Bounded queue: maxsize=64 caps how many produced items
                # can sit ahead of the consumer, giving backpressure on
                # the producer thread pool.
                q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
                # Fresh unique sentinel per batch (a bare `object()`, so
                # `is` comparison in `_consumer` can never false-positive
                # match a real dict work item).
                sentinel = object()
                consumer_done = threading.Event()

                collected_preds: List[gpd.GeoDataFrame] = []
                lock = threading.Lock()

                # Single consumer thread per batch: serializes all
                # GPU/Keras/PyTorch work for this batch's tiles.
                consumer_thread = threading.Thread(
                    target=self._consumer,
                    args=(q, sentinel, collected_preds, fails,
                          consumer_done, lock),
                    daemon=True
                )
                consumer_thread.start()

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    # Kick off all producers for this batch concurrently
                    # (I/O-bound GEE downloads / cache reads).
                    future_to_tile = {
                        ex.submit(self.produce_tile_input, tile):
                        tile for tile in batch_tiles
                    }
                    for fut in as_completed(future_to_tile):
                        tile = future_to_tile[fut]
                        try:
                            item = fut.result()
                            # Hand the produced item to the consumer;
                            # blocks here if the queue is at maxsize=64
                            # (backpressure).
                            q.put(item)
                        except Exception as e:
                            self.logger.error(
                                f"Producer failed for tile {tile.key}: {e}")
                            fails.append(tile)

                # Signal "no more work" and wait for the consumer to
                # finish draining the queue before merging results.
                q.put(sentinel)
                consumer_done.wait()

                if collected_preds:
                    # Merge this batch's per-tile GeoDataFrames into one.
                    batch_gdf = self._ensure_gdf(
                        gpd.pd.concat(collected_preds, ignore_index=True))
                    batch_gdf = batch_gdf.set_crs(
                        'epsg:4326', allow_override=True)
                    predictions = predictions.set_crs(
                        'epsg:4326', allow_override=True)
                    # Append this batch onto the running cumulative total.
                    predictions = self._ensure_gdf(
                        gpd.pd.concat([predictions, batch_gdf],
                                      ignore_index=True))
                    self.logger.info(f"Found {len(batch_gdf)} new positives.")
                    print(f"Found {len(batch_gdf)} new positives.", flush=True)

                    # Checkpoint: rewrite the FULL cumulative GeoJSON
                    # after every batch, so a crash mid-run costs at most
                    # the current batch's work, not the whole run.
                    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_file(outpath, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            # Next pass (if any tries remain) retries only the failures.
            retry_tiles = fails
            tries_remaining -= 1

        return predictions

    def predict_on_tile(self, tile: TileType) -> gpd.GeoDataFrame:
        """
        Convenience wrapper for debugging: run full inference on a single tile.
        Reuses the same logic as bulk_predict but without producer/consumer.

        Used e.g. from inference.ipynb for interactive single-tile
        debugging, where the producer/consumer machinery of bulk_predict
        would be unnecessary overhead.

        Args:
            tile: TileType, the single tile to run inference on.

        Returns:
            gpd.GeoDataFrame: the positive-chip detections for this tile
            (possibly empty); unlike `predict_on_tile_pixels`, this
            method does NOT return a (gdf, failed_tile) tuple — any
            failure would propagate as a raised exception instead, since
            there is no batch-retry mechanism here.

        Side effects: same as `predict_on_tile_pixels`/
        `predict_on_tile_embeddings` (possible GEE network call or cache
        read, embedding backbone forward pass, Keras predict, inline SAM2
        masking with GeoTIFF writes).
        """
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is not None:
            emb_path = self._make_embedding_cache_path(tile)
            if emb_path.exists():
                # Fast path: classify straight from cached embeddings,
                # skipping pixel download entirely.
                gdf = gpd.read_parquet(emb_path)
                chip_geoms = gdf[["geometry"]].copy()
                embeddings = gdf.drop(
                    columns="geometry",
                    errors="ignore").to_numpy(dtype=np.float32)
                preds_gdf, _ = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                return preds_gdf

        # No usable embeddings cache: fetch pixels and run the full
        # pixel-to-prediction pipeline (including optional SAM2 masking).
        pixels = self.data_extractor.get_tile_data(tile)
        preds_gdf, _ = self.predict_on_tile_pixels(pixels, tile)
        return preds_gdf

class SAM2_Masker:
    """Computes pixelwise segmentations."""
    def __init__(
        self, data_extractor: GEE_Data_Extractor, config: MaskConfig):
        """Select a compute device and load the fine-tuned SAM2 model.

        Args:
            data_extractor: GEE_Data_Extractor, reused for tile pixel
                fetches (`bulk_mask_polygons`) and for `affine_from_tile`
                / `save_tile` georeferencing helpers.
            config: MaskConfig, SAM2 checkpoint/config paths and
                spatial-prior/smoothing parameters.

        Side effects: loads SAM2 model weights from disk (in
        `_load_model`, called at the end of this constructor); creates a
        threading.Lock to serialize the (stateful) SAM2 predictor's
        set_image/predict call pairs across threads.

        NOTE: unlike InferenceEngine.__init__, this constructor has NO
        `mps` branch — only `cuda` else `cpu`. On Apple Silicon Macs this
        means SAM2 always runs on CPU (slower), even though the rest of
        the pipeline (InferenceEngine) would use the `mps` GPU backend if
        available.
        """
        self.data_extractor = data_extractor
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # No `mps` branch here (see NOTE in docstring above): Macs
            # fall straight through to CPU for SAM2, unlike
            # InferenceEngine's device selection.
            self.device = torch.device("cpu")

        # SAM2ImagePredictor is stateful (set_image() then predict()
        # must not be interleaved across threads), so this lock
        # serializes access to `self.predictor` across concurrent callers.
        self._sam_lock = threading.Lock()

        self.predictor = self._load_model()

    def _load_model(self):
        """Build the base SAM2 model, overlay fine-tuned weights, and wrap it.

        No inputs beyond `self` (reads `self.config` and `self.device`).

        Returns:
            sam2.sam2_image_predictor.SAM2ImagePredictor wrapping the
            loaded model, ready for `.set_image()` / `.predict()` calls.

        Side effects: imports the `sam2` package lazily (only needed
        when SAM2 masking is actually used, avoiding a hard dependency
        for pipeline runs that don't use it); reads both the base SAM2
        checkpoint and the fine-tuned weights file from disk.
        """
        # Lazy import: the sam2 package (and its dependencies) is only
        # required when SAM2 masking is actually used.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Builds the base architecture and loads the (non-fine-tuned)
        # base checkpoint, via Hydra config composition
        # (self.config.sam2_model_cfg must be a package-relative config
        # name, not a filesystem path — see DEFAULT_SAM2_HYDRA_CONFIG).
        sam2_model = build_sam2(
            self.config.sam2_model_cfg,
            self.config.sam2_checkpoint,
            device=self.device)

        # Overlay the AMW fine-tuned weights on top of the base model.
        # map_location="cpu" loads tensors to CPU memory first regardless
        # of `self.device` (they get moved to the right device as part
        # of build_sam2 already having placed the model there;
        # load_state_dict then copies matching parameters in place).
        state = torch.load(self.config.finetuned_weights, map_location="cpu")
        # strict=False: tolerate key mismatches between the fine-tuned
        # state_dict and the base model's parameters (e.g. if fine-tuning
        # only touched a subset of layers).
        sam2_model.load_state_dict(state, strict=False)
        # eval() mode: disable dropout/training-time batchnorm behavior.
        sam2_model.eval()

        return SAM2ImagePredictor(sam2_model)

    @staticmethod
    def polygon_gdf_to_pixel_bbox(
        gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> Optional[np.ndarray]:
        """Convert georeferenced polygons to a box prompt in pixel coords.

        Collapses ALL polygons in `gdf` into a single bounding box prompt
        (SAM2 here is given one box per tile, not one box per detected
        chip).

        Args:
            gdf: gpd.GeoDataFrame of detection polygons (chip footprints)
                in the same CRS the transform expects (EPSG:4326, lon/lat
                degrees, matching `transform`'s domain).
            transform: affine.Affine mapping pixel (col, row) -> (lon,
                lat); its inverse (`~transform`) maps (lon, lat) -> pixel
                (col, row).
            width: int, raster width in pixels (used to clip the box to
                valid pixel bounds).
            height: int, raster height in pixels (used to clip the box).

        Returns:
            Optional[np.ndarray], dtype int32, shape (4,): [x0, y0, x1,
            y1] pixel-coordinate box (x=column, y=row), clipped to
            [0, width-1] / [0, height-1] — the format SAM2's `predict(
            box=...)` expects. Returns None if `gdf` is empty, its union
            geometry is empty, or the resulting box is degenerate
            (zero or negative width/height after clipping).

        Side effects: none (pure function).
        """
        if gdf.empty:
            return None

        # Merge all detection polygons in this tile into one geometry so
        # a single overall bounding box can be computed.
        geom = gdf.geometry.union_all()
        if geom.is_empty:
            return None

        # bounds in lon/lat degrees: (min lon, min lat, max lon, max lat).
        minx, miny, maxx, maxy = geom.bounds
        # Inverse affine: maps geographic (lon, lat) -> pixel (col, row).
        inv = ~transform

        # NOTE on the y-axis flip: in image/pixel space, row increases
        # downward while latitude increases upward, so the NW (top-left)
        # geographic corner is (minx, maxy), and the SE (bottom-right)
        # corner is (maxx, miny) — NOT (minx, miny)/(maxx, maxy).
        x0, y0 = inv * (minx, maxy)  # NW corner -> top-left pixel
        x1, y1 = inv * (maxx, miny)  # SE corner -> bottom-right pixel

        # floor/ceil then clip to valid pixel index range, so the box
        # never references out-of-bounds pixels even if geometry extends
        # slightly beyond the tile (e.g. due to rounding).
        x0 = int(np.clip(np.floor(x0), 0, width - 1))
        x1 = int(np.clip(np.ceil(x1), 0, width - 1))
        y0 = int(np.clip(np.floor(y0), 0, height - 1))
        y1 = int(np.clip(np.ceil(y1), 0, height - 1))

        if x1 <= x0 or y1 <= y0:
            # Degenerate box (can happen after aggressive clipping):
            # nothing sensible to prompt SAM2 with.
            return None

        return np.array([x0, y0, x1, y1], dtype=np.int32)

    def get_rgb(self, pixels: np.ndarray) -> np.ndarray:
        """Convert Sentinel-2 reflectance -> uint8 RGB w/ constrast stretch.

        Args:
            pixels: np.ndarray, shape (H, W, bands), raw digital-number
                values in the 13-band S2L1C band order (see
                DataConfig._BAND_IDS["S2L1C"]); indices [3, 2, 1]
                correspond to bands B4 (red), B3 (green), B2 (blue).

        Returns:
            np.ndarray, dtype uint8, shape (H, W, 3) — an ordinary RGB
            image (as SAM2's image predictor expects), with a fixed
            /3000 contrast stretch: raw DN/3000 approximates a
            reflectance-to-display mapping where ~0.30 reflectance maps
            to saturated white, a standard "bright earth" visualization
            stretch (not a physically calibrated reflectance value).

        Side effects: none (pure function).
        """
        # Select bands [B4, B3, B2] = [red, green, blue] from the 13-band
        # stack and drop to float32 for the division below.
        rgb = pixels[..., [3, 2, 1]].astype(np.float32)
        # Fixed contrast stretch: divide by 3000 (an empirically chosen
        # "typical bright" DN) and clip to [0, 1] before scaling to uint8.
        rgb = np.clip(rgb / 3000.0, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def soft_spatial_prior(
        self, preds_gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> np.ndarray:
        """
        Piecewise spatial logit prior:
        - Inside detection mask: 0
        - Outside: quadratic negative penalty increasing with distance

        Args:
            preds_gdf: gpd.GeoDataFrame of classifier-positive chip
                geometries for this tile (EPSG:4326), used to rasterize
                the "inside detection" region.
            transform: affine.Affine, pixel<->geographic transform for
                this tile (same one used to fetch/georeference `pixels`).
            width: int, raster width in pixels.
            height: int, raster height in pixels.

        Returns:
            np.ndarray, dtype float32, shape (height, width): a spatial
            LOG-ODDS prior added to SAM2's raw logits later
            (`SAM2_Masker.predict`). Value is exactly 0.0 for pixels
            inside any detection polygon, and
            -(distance_in_px / prior_sigma)^2 for pixels outside — a
            quadratic (in Euclidean pixel distance) negative penalty
            that grows the farther a pixel is from the nearest
            classifier detection. This discourages SAM2 from extending
            its segmentation mask to distant look-alike features (e.g.
            rivers, bare soil) that were never flagged by the classifier.

        Side effects: none (pure function; the rasterize call reads only
        `preds_gdf`/`transform`, no external I/O).
        """
        # Burn all (non-null, non-empty) detection polygons into a
        # boolean raster: True where "inside a detection", matching the
        # tile's pixel grid via `transform`.
        raster = rasterio.features.rasterize(
            ((geom, 1) for geom in preds_gdf.geometry
            if geom is not None and not geom.is_empty),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        # Euclidean distance transform (in pixels) computed over the
        # INVERSE mask (~raster): for every "outside" pixel, the distance
        # to the nearest "inside" (True) pixel; "inside" pixels get 0
        # from this call, but they get set to 0 explicitly below anyway.
        dist_outside = ndi.distance_transform_edt(~raster)

        # Quadratic decay: penalty grows more severe (more negative) the
        # farther outside a detection a pixel is, scaled by prior_sigma
        # (larger sigma = more permissive/slower falloff).
        penalty = -(dist_outside / self.config.prior_sigma) ** 2
        prior = np.zeros_like(penalty, dtype=np.float32)
        # Only apply the penalty outside detections; inside stays 0
        # (i.e. no prior adjustment there).
        prior[~raster] = penalty[~raster]

        return prior

    @staticmethod
    def resize_prior_to_logits(
        prior: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample prior field to SAM2 logit resolution.

        Args:
            prior: np.ndarray, shape (H, W) at full tile resolution (as
                produced by `soft_spatial_prior`).
            target_shape: Tuple[int, int], (H_logits, W_logits) — SAM2's
                internal (lower-resolution) logits grid shape, which is
                coarser than the full tile raster.

        Returns:
            np.ndarray, dtype float32, shape `target_shape` — `prior`
            resampled (bilinearly) down to the logits grid so it can be
            added directly to SAM2's raw logits.

        Side effects: none (pure function).
        """
        # Per-axis zoom factors to go from `prior`'s current resolution
        # to the target (SAM2 logits) resolution.
        scale_y = target_shape[0] / prior.shape[0]
        scale_x = target_shape[1] / prior.shape[1]

        # order=1: bilinear interpolation (smooth resampling, appropriate
        # for a continuous-valued prior field rather than a categorical
        # mask).
        out = ndi.zoom(prior, (scale_y, scale_x), order=1)
        return out.astype(np.float32)

    def upsample_logits(
        self, logits: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample SAM2 logits to target raster resolution, with optional
            Gaussian smoothing for spatial regularization.
        """
        logits_tensor = torch.from_numpy(logits[None, None, ...]).float()
        upsampled = F.interpolate(
            logits_tensor,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )[0,0].numpy()

        sigma = self.config.smoothing_sigma
        if sigma and sigma > 0:
            upsampled = ndi.gaussian_filter(upsampled, sigma=sigma)

        return upsampled

    def predict(self, pixels: np.ndarray, tile: TileType,
                preds_gdf: gpd.GeoDataFrame):
        """Run SAM2 using polygon-derived box prompts.

        pixels: np.ndarray: reflectance values (H, W, B)
        tile: TileType
        preds_gdf: GeoDataFrame in EPSG:4326 (positive chip geometries).

        Full step-by-step behavior:
          1. Build a georeferencing transform for this tile.
          2. Collapse `preds_gdf`'s chip geometries into ONE bounding-box
             prompt in pixel coordinates (`polygon_gdf_to_pixel_bbox`);
             bail out early (return {}) if no valid box can be built.
          3. Under `self._sam_lock` (SAM2's predictor is stateful and not
             thread-safe): set the tile's RGB image on the predictor,
             then run SAM2's box-prompted prediction, requesting 3
             candidate masks (`multimask_output=True`) with associated
             quality scores and raw logits.
          4. Take the logits of the highest-scoring candidate mask.
          5. Compute the classifier-derived spatial prior (see
             `soft_spatial_prior`), resize it to the logits' (lower)
             resolution, and ADD it to the SAM2 logits in log-odds space
             — this is how the classifier's chip-level detections steer/
             constrain SAM2's pixel-level segmentation.
          6. Upsample the combined log-odds field back to full tile
             resolution (with optional Gaussian smoothing).
          7. Threshold at 0 (log-odds 0 corresponds to probability 0.5)
             to get the final binary uint8 mask.
          8. Write both the binary mask and the raw combined log-odds
             field to disk as separate single-band GeoTIFFs.

        Returns:
            Dict[str, Any]: either `{}` (no valid box prompt could be
            built, e.g. `preds_gdf` was empty) or a diagnostics dict:
            {"labels": ..., "scores": ..., "prob_logits": ...,
             "box_prompt": np.ndarray[4] int32} — SAM2's raw
            multi-mask outputs plus the box prompt used, for inspection/
            debugging (not persisted to disk itself).

        Side effects:
            - Acquires `self._sam_lock` for the duration of the SAM2
              set_image/predict call pair.
            - Writes two GeoTIFFs to `self.config.mask_dir` via
              `self.data_extractor.save_tile`: the binary mask
              (product_type="mask", suffix "-msk") and the combined
              log-odds field (product_type="logits", suffix "-logits").
        """
        height, width = pixels.shape[:2]
        transform = self.data_extractor.affine_from_tile(tile, width, height)

        box_prompt = self.polygon_gdf_to_pixel_bbox(
            preds_gdf, transform, width, height)

        if box_prompt is None:
            # No usable detections to build a box from: nothing to
            # segment for this tile.
            return {}

        with self._sam_lock:
            # set_image + predict must be called as a matched pair on
            # the same (stateful) predictor instance without another
            # thread interleaving a different image in between.
            self.predictor.set_image(self.get_rgb(pixels))
            # multimask_output=True: SAM2 returns 3 candidate masks (with
            # per-mask quality `scores`) instead of a single best guess,
            # since a single box prompt can be ambiguous.
            labels, scores, prob_logits = self.predictor.predict(
                box=box_prompt,
                multimask_output=True)

        # Pick the highest-quality-scoring candidate mask's raw logits.
        best_logits = prob_logits[np.argmax(scores)]

        # Compute the classifier-detection-based spatial prior at full
        # tile resolution, then resample it down to match SAM2's
        # (coarser) logits grid so the two can be added directly.
        prior = self.soft_spatial_prior(preds_gdf, transform, width, height)
        prior = self.resize_prior_to_logits(prior, best_logits.shape)
        # Combine in LOG-ODDS space: adding the prior nudges probability
        # toward 0 (i.e. "not mine") the farther a pixel is from a
        # classifier detection, without needing to convert to/from
        # probability space first.
        log_odds = best_logits + prior
        # Upsample the combined field back to full tile pixel resolution
        # (with optional Gaussian smoothing of the boundary).
        upsampled = self.upsample_logits(log_odds, pixels.shape[:2])
        # log-odds > 0 <=> probability > 0.5: threshold to a binary mask.
        mask = (upsampled > 0).astype('uint8')

        self.data_extractor.save_tile(
            pixels=mask[..., None],  # single band
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="mask")

        self.data_extractor.save_tile(
            pixels=log_odds[..., None],
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="logits")

        return {
            "labels": labels,
            "scores": scores,
            "prob_logits": prob_logits,
            "box_prompt": box_prompt,
        }

    def _simplify_for_tiling(
        self, gdf: gpd.GeoDataFrame,
        buffer_width: float = 0.005) -> gpd.GeoDataFrame:
        """Simplify polygons for tile creation.

        Argument buffer_width defines the smoothing scale, in degrees
            for epsg:4326 geometries.

        Used ONLY to build the tiling grid footprint in
        `bulk_mask_polygons` (i.e. to decide WHICH tiles need to be
        fetched/processed) — the original, unsimplified `polys_gdf` is
        still what is actually clipped per-tile and passed to `predict`
        for the real segmentation work.

        Args:
            gdf: gpd.GeoDataFrame of input polygons (EPSG:4326).
            buffer_width: float, degrees; the morphological closing
                (dilate then erode) scale used to fuse nearby fragments
                and smooth jagged boundaries. Default 0.005 degrees
                (~500 m at the equator).

        Returns:
            gpd.GeoDataFrame: simplified geometries — holes removed
            (only exterior rings kept), invalid geometries repaired via
            zero-width buffering, and rows with invalid/empty resulting
            geometry dropped, index reset.

        Side effects: none (returns a new GeoDataFrame; does not mutate
        the input `gdf`, since `.copy()` is used up front).
        """
        gdf = gdf.copy()

        # Morphological "closing": buffer outward then inward by the same
        # amount. This fuses geometries that are close together (within
        # buffer_width) into single shapes and smooths small notches —
        # producing a cleaner footprint to build a tile grid over,
        # without needing every tiny fragment to get its own tile.
        # join_style=2 (mitre) is used for the buffer corner style.
        gdf["geometry"] = gdf.geometry.buffer(
            buffer_width, join_style=2).buffer(
            -buffer_width, join_style=2)

        def _drop_holes(geom):
            # Keep only the exterior ring of each polygon (discard
            # interior holes) — for tiling-grid purposes, holes just add
            # unnecessary complexity without changing which tiles are
            # needed to cover the shape.
            if geom.is_empty or geom is None:
                return None
            if geom.geom_type == "Polygon":
                return type(geom)(geom.exterior)
            elif geom.geom_type == "MultiPolygon":
                return type(geom)([type(p)(p.exterior) for p in geom.geoms
                                       if not p.is_empty])
            else:
                return geom

        gdf["geometry"] = gdf.geometry.map(_drop_holes)

        with warnings.catch_warnings():
            # buffer(0) is a common shapely idiom to repair minor
            # self-intersections/invalid geometries; it can emit noisy
            # UserWarnings that are safe to ignore here.
            warnings.simplefilter("ignore", UserWarning)
            gdf["geometry"] = gdf.buffer(0)

        # Drop anything that is still invalid or became empty after the
        # repair step above.
        gdf = gdf[gdf.is_valid & ~gdf.is_empty].reset_index(drop=True)

        return gdf

    def bulk_mask_polygons(
        self, polys_gdf: gpd.GeoDataFrame,
        max_concurrent_tiles=500) -> gpd.GeoDataFrame:
        """Compute segmentation masks for wide-area polygons.

        Standalone (non-inline) SAM2 masking entry point: given an
        arbitrary set of already-known detection polygons (e.g. loaded
        from a previously saved predictions GeoJSON), builds a tile grid
        covering them and runs SAM2 segmentation over every tile that
        actually intersects a polygon.

        Args:
            polys_gdf: gpd.GeoDataFrame of detection polygons, any CRS
                (re-projected to EPSG:4326 internally).
            max_concurrent_tiles: int, tiles processed per
                ThreadPoolExecutor batch (memory-bounding, same rationale
                as InferenceEngine.bulk_predict). Default 500.

        Returns:
            gpd.GeoDataFrame: NOTE — despite the return type annotation,
            this method's body has no explicit `return` statement for a
            result GeoDataFrame; it always implicitly returns None. Mask
            and logits outputs are instead written directly to disk (via
            `self.predict` -> `self.data_extractor.save_tile`) as a side
            effect, not accumulated/returned in memory. (Documented here
            as observed behavior; not fixed since this is a doc-only
            annotation pass.)

        Side effects:
            - Builds a full DLTile/CenteredTile grid over (a simplified
              version of) the input polygons' combined footprint.
            - For every tile that intersects at least one polygon: fetches
              tile pixels via GEE (network call / cache read) and runs
              SAM2 segmentation, writing mask + logits GeoTIFFs to disk.
            - Prints progress/failure messages to stdout; per-tile
              failures are caught and printed, NOT retried (unlike
              InferenceEngine.bulk_predict, there is no retry-pass logic
              here).
        """

        polys_gdf = polys_gdf.to_crs("EPSG:4326")
        # Build a simplified footprint purely to lay out the tile grid
        # (see _simplify_for_tiling docstring) — the tiling grid itself
        # doesn't need exact polygon boundaries, just decent coverage.
        region_gdf = self._simplify_for_tiling(polys_gdf)
        tiles = create_tiles(
            region_gdf.union_all(),
            self.data_extractor.config.tilesize,
            self.data_extractor.config.pad
        )
        print(f'{len(tiles)} tiles created.')

        # --- precompute intersections single-threaded - not thread safe ---
        # gpd.clip is not thread-safe, so this pass (finding which
        # polygons fall in which tile) must run single-threaded, ahead of
        # the concurrent SAM2-processing pass below.
        tiles_w_polys = []
        for tile in tqdm(tiles, desc="Clipping polygons"):
            # Clip the ORIGINAL (unsimplified) polygons to this tile's
            # footprint — the real geometry used for the box prompt and
            # spatial prior in `predict`.
            tile_polys = gpd.clip(polys_gdf, tile.geometry)
            tile_polys = tile_polys[
                tile_polys.is_valid &
                ~tile_polys.geometry.is_empty
            ]
            if not tile_polys.empty:
                # Only tiles that actually contain a valid, non-empty
                # clipped polygon are worth running SAM2 on.
                tiles_w_polys.append((tile, tile_polys))

        # --- SAM2 masking multi-threaded ---
        def process_tile(tile: TileType, tile_polys: gpd.GeoDataFrame):
            # Fetch pixels (network/cache I/O) then run SAM2 for this
            # single tile using its pre-clipped polygons.
            pixels = self.data_extractor.get_tile_data(tile)
            return self.predict(pixels, tile, tile_polys)

        for i in tqdm(range(0, len(tiles_w_polys), max_concurrent_tiles),
                      desc="Processing tiles"):
            batch = tiles_w_polys[i : i + max_concurrent_tiles]

            with ThreadPoolExecutor(
                max_workers=self.data_extractor.config.max_workers) as ex:
                futures = {
                    ex.submit(process_tile, tile, tp): tile for tile, tp
                        in batch
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        # Failures are logged but NOT retried (unlike
                        # InferenceEngine.bulk_predict's tries/retry_tiles
                        # loop) — a failed tile here is simply skipped.
                        print(f"Tile failed with error: {e}", flush=True)

