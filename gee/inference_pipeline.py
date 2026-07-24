# =============================================================================
# FILE OVERVIEW: inference_pipeline.py
#
# PURPOSE
#   Command-line wrapper around the inference engine defined in gee.py. It
#   turns a region-of-interest (GeoJSON polygon(s)) into a set of predicted
#   "mining chip" detections for a given date range, using the Sentinel-2 (or
#   other GEE collection) imagery pipeline, the SSL4EO ViT-S/16 embedding
#   backbone, and a Keras MLP ensemble classifier head. Every command-line
#   flag mirrors a field of one of the three gee.py dataclasses
#   (DataConfig / InferenceConfig / MaskConfig); those dataclasses are the
#   single source of truth for defaults and validation, and this file is a
#   thin CLI "skin" on top of them (see the `config_dict = {...}` pattern in
#   `__main__` below, which rebuilds each dataclass from parsed argparse
#   values by matching field names).
#
# INVOCATION
#   Run from inside the gee/ directory (relative default paths, e.g. the
#   default Keras model path and default region path, assume this cwd):
#       python inference_pipeline.py --region_path ../data/boundaries/foo.geojson \
#           --start_date 2023-01-01 --end_date 2023-12-31 [other flags...]
#   Importing gee (see gee.py module-level code) requires Google Earth Engine
#   credentials to already be configured on the machine (Application Default
#   Credentials via `gcloud auth application-default login`), because gee.py
#   calls ee.Initialize() at import time.
#
# ON-DISK INPUTS
#   - --region_path: a GeoJSON/vector file (any geopandas-readable format)
#     defining the region(s) of interest, in any CRS geopandas can parse.
#   - --model_path: a Keras .h5 classifier (ensemble MLP head).
#   - Optional: --image_cache_dir, --embeddings_cache_dir for caching
#     GEE-downloaded rasters / SSL4EO embeddings across runs.
#   - Optional SAM2 checkpoint/config paths if --run_sam2 is set.
#
# ON-DISK OUTPUTS
#   - A rolling log file at <logdir>/gee_<region filename>.log (rotating,
#     1 MB per file, 5 backups kept).
#   - A GeoJSON of positive-chip detections written by gee.InferenceEngine
#     to data/outputs/<model_version>/<region>_<model_version>_<thr>_
#     <start>_<end>.geojson (path and naming logic lives in gee.py's
#     InferenceEngine.predictions_geojson_path / bulk_predict).
#
# PIPELINE POSITION
#   This script is the top-level entry point for BULK inference over a
#   region: region -> DLTile grid tiles (tile_utils.create_tiles) -> GEE
#   composite fetch -> chip cut -> SSL4EO embed -> Keras ensemble predict ->
#   (optional) SAM2 segmentation -> GeoJSON of detections. The heavy lifting
#   (GEE_Data_Extractor, InferenceEngine, SAM2_Masker) all lives in gee.py;
#   this file only parses CLI args, builds the three config dataclasses,
#   sets up logging, and calls into that engine.
# =============================================================================

import argparse
from dataclasses import fields
from datetime import datetime
import logging
import logging.handlers
import re
from pathlib import Path

import geopandas as gpd

# Local modules: gee.py holds the GEE/ML inference engine and its config
# dataclasses; tile_utils.py holds the DLTile-based tiling grid helpers.
import gee
import tile_utils

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged.

    Used as an argparse `type=` callable so the parser itself rejects
    malformed dates before any GEE calls are made.

    Args:
        s: candidate date string from the command line, e.g. "2023-01-01".

    Returns:
        The same string, unchanged, if it matches YYYY-MM-DD.

    Raises:
        argparse.ArgumentTypeError: if `s` does not match the pattern; this
        is the exception type argparse expects so it can print a clean
        usage error instead of a raw traceback.

    Side effects: none (pure validation function).
    """
    # Anchor the regex to the full string (^...$) so partial/garbled dates
    # like "2023-01-01x" are rejected, not silently accepted.
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def get_logger(logpath: Path, maxBytes=1e6,
               backupCount=5, level=logging.INFO) -> logging.Logger:
    """Configure and return the root logger with a rotating file handler.

    Args:
        logpath: Path to the log file to write to; parent directories are
            created if missing.
        maxBytes: maximum size in bytes of a single log file before it is
            rotated (default 1e6 = 1 MB).
        backupCount: number of rotated backup files to keep (default 5,
            i.e. up to 5 old logs plus the current one).
        level: minimum logging.Level to emit (default INFO).

    Returns:
        The configured `logging.Logger` (the root logger — note this
        mutates global logging state by attaching a handler to it, so
        calling this twice in one process would attach duplicate handlers).

    Side effects:
        - Creates `logpath.parent` on disk if it does not exist.
        - Creates/opens the rotating log file at `logpath`.
        - Mutates the process-wide root logger (adds a handler, sets level).
    """
    # Ensure the log directory exists before the handler tries to open a
    # file inside it (RotatingFileHandler does not create parent dirs).
    logpath.parent.mkdir(parents=True, exist_ok=True)
    # RotatingFileHandler auto-rotates to logpath.1, logpath.2, ... once
    # maxBytes is exceeded, keeping at most backupCount old files.
    handler = logging.handlers.RotatingFileHandler(
        logpath, maxBytes=maxBytes, backupCount=backupCount
    )
    # getLogger() with no name returns the root logger, so this affects
    # logging.info(...) calls anywhere in the process, not just this module.
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def main(data_config: gee.DataConfig,
         inference_config: gee.InferenceConfig,
         mask_config: gee.MaskConfig,
         cli_args: argparse.Namespace,
         logger: logging.Logger):
    """Run bulk inference over a region and log a summary of results.

    Args:
        data_config: gee.DataConfig — GEE collection/tiling/download
            settings (tile size in px, cloud-clear threshold, etc.).
        inference_config: gee.InferenceConfig — model path, embedding
            backbone settings, prediction threshold, concurrency settings.
        mask_config: gee.MaskConfig — SAM2 segmentation settings (only
            used if inference_config.run_sam2 is True).
        cli_args: parsed argparse.Namespace; specifically used here for
            region_path, start_date, and end_date (the "general" args that
            are not part of any of the three dataclasses above).
        logger: logging.Logger to write progress/summary messages to.

    Returns: None.

    Side effects:
        - Reads the region file from disk (cli_args.region_path).
        - Triggers many outbound GEE API calls (imagery downloads) and
          local model inference (via engine.bulk_predict).
        - Writes the cumulative predictions GeoJSON to disk (inside
          engine.bulk_predict, checkpointed after every tile batch).
        - Writes several INFO-level log lines summarizing the run.

    Pipeline role: this is the single orchestration function tying region
    geometry -> tile grid -> InferenceEngine -> GeoJSON output together.
    """
    # Read the region file (any geopandas-supported vector format) and
    # union all its features into one (possibly multi-part) geometry, so
    # multi-polygon regions are tiled as a single continuous area.
    region = gpd.read_file(cli_args.region_path).union_all()

    # Cover the unioned region geometry with a grid of DLTile tiles sized
    # data_config.tilesize px (+ data_config.pad px of context per side).
    # Tiles come from the Descartes Labs UTM tiling grid (see tile_utils.py).
    tiles = tile_utils.create_tiles(
        region, data_config.tilesize, data_config.pad)
    logger.info(f"Created {len(tiles)} tiles")

    # Build the inference engine: this loads the Keras classifier and (if
    # configured) the SSL4EO embedding backbone and SAM2 masker, and lazily
    # builds (but does not yet execute) the GEE composite recipe for the
    # given date range and collection.
    engine = gee.InferenceEngine(
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
        data_config=data_config,
        config=inference_config,
        mask_config=mask_config,
        logger=logger,
    )
    # Run the full pipeline over every tile: fetch pixels -> cut chips ->
    # embed -> classify -> (optional) SAM2 mask -> write GeoJSON. Returns
    # a GeoDataFrame of all chips whose confidence exceeded pred_threshold.
    preds = engine.bulk_predict(tiles, cli_args.region_path.stem)

    # KNOWN BUG (log-only, intentionally not fixed here): tiles are
    # data_config.tilesize px square at ~10 m/px, so a 576-px tile spans
    # 5.76 km per side = 33.18 km^2 (~3318 ha), not (tilesize/100)^2 ha.
    # This formula actually computes area in km^2 (since 576/100 = 5.76 =
    # the tile's side length in km), but the units are then mislabeled
    # "ha" in the log line below. The mistake is cosmetic/log-only: it
    # does not affect tiling, inference, or the saved GeoJSON, only this
    # printed summary statistic (off by a factor of 100).
    analyzed_area = len(tiles) * (data_config.tilesize / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above "
                f"{inference_config.pred_threshold}")


if __name__ == '__main__':
    # Instantiate each config dataclass with its built-in defaults purely
    # to read those defaults back out for use as argparse `default=` values
    # (keeps the CLI defaults in sync with the dataclass defaults instead
    # of duplicating literal values here).
    data_defaults = gee.DataConfig()
    inference_defaults = gee.InferenceConfig()
    mask_defaults = gee.MaskConfig()

    parser = argparse.ArgumentParser(description="Run bulk ML inference.")

    # ---- DataConfig args: GEE collection, tiling, and download settings ----
    parser.add_argument("--tilesize", type=int,
                        default=data_defaults.tilesize,
                        help="Tile width in pixels for requests to GEE")
    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Number of pixels to pad each tile")
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite image collection")
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Maximum concurrent GEE requests")
    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional directory to save/reload image rasters")


    # ---- InferenceConfig args: model, embedding backbone, concurrency ----
    parser.add_argument("--model_path", type=Path,
                        default=inference_defaults.model_path,
                        help="Path to saved Keras classifier (.h5)")
    parser.add_argument("--pred_threshold", type=float,
                        default=inference_defaults.pred_threshold,
                        help="Prediction threshold for positive chips")
    parser.add_argument("--stride_ratio", type=int,
                        default=inference_defaults.stride_ratio,
                        help="Stride is computed as chip_size//stride_ratio")
    parser.add_argument("--max_concurrent_tiles", type=int,
                        default=inference_defaults.max_concurrent_tiles,
                        help="Maximum number of tiles to process concurrently")
    parser.add_argument("--tries", type=int,
                        default=inference_defaults.tries,
                        help="Number of retries per tile")
    parser.add_argument("--embed_model_name", type=str,
                        default=inference_defaults.embed_model_name,
                        help="Embedding model identifier")
    parser.add_argument("--embed_model_path", type=str,
                        default=inference_defaults.embed_model_path,
                        help=("Path to optional pretrained foundation model; "
                              "set '' to disable"))
    parser.add_argument("--embed_model_chip_size", type=int,
                        default=inference_defaults.embed_model_chip_size,
                        help="Input size for embedding model")
    parser.add_argument("--embedding_batch_size", type=int,
                        default=inference_defaults.embedding_batch_size,
                        help="Batch size for embedding model inference")
    parser.add_argument("--geo_chip_size", type=int,
                        default=inference_defaults.geo_chip_size,
                        help="Input size for embedding model")
    parser.add_argument("--embeddings_cache_dir", type=str,
                        default=inference_defaults.embeddings_cache_dir,
                        help=("Optional directory to save/reload embeddings. "
                              "Tiles read from cache skip fetching pixels; "
                              "with --run_sam2, inline SAM2 does not run for "
                              "those tiles (use standalone SAM2 masking if needed)."))
    parser.add_argument("--run_sam2", action="store_true",
                        default=inference_defaults.run_sam2,
                        help="Enable SAM2 segmentation after model predictions")
    parser.add_argument(
        "--inference_output_base",
        type=str,
        default=str(inference_defaults.inference_output_base),
        help=(
            "Base directory for prediction GeoJSON outputs "
            "(default: <repo>/data/outputs; subfolder per model version)"
        ),
    )

    # ---- MaskConfig args: SAM2 segmentation settings (used only if --run_sam2) ----
    parser.add_argument("--prior_sigma", type=float,
                        default=mask_defaults.prior_sigma,
                        help="Spatial prior falloff (pixels)")
    parser.add_argument("--smoothing_sigma", type=float,
                        default=mask_defaults.smoothing_sigma,
                        help="For Gaussian smoothing (pixels)")
    parser.add_argument("--sam2_repo_path", type=str,
                        default=mask_defaults.sam2_repo_path,
                        help="Path to SAM2 repository root")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default=mask_defaults.sam2_checkpoint,
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--finetuned_weights", type=str,
                        default=mask_defaults.finetuned_weights,
                        help="Path to fine-tuned SAM2 model weights")
    parser.add_argument("--sam2_model_cfg", type=str,
                        default=mask_defaults.sam2_model_cfg,
                        help="Hydra config name for SAM2 (e.g. configs/sam2.1/"
                             "sam2.1_hiera_s.yaml), or path to a YAML under "
                             "sam2_repo/sam2/configs/")
    parser.add_argument("--mask_dir", type=str,
                        default=mask_defaults.mask_dir,
                        help="Directory to save SAM2 outputs")

    # ---- General args: not part of any config dataclass ----
    parser.add_argument("--region_path", type=Path,
                        default="../data/boundaries/amazon_basin.geojson",
                        help="Path to ROI geojson")
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--logdir", type=Path,
                        default=Path("../logs"),
                        help="Directory for log files")

    args = parser.parse_args()

    # Rebuild DataConfig from the parsed CLI namespace: for every dataclass
    # field name, pull the same-named attribute off `args` (None if
    # somehow missing). This is the CLI-args-are-a-skin-over-dataclasses
    # pattern referenced in the file overview above — it means every field
    # added to DataConfig automatically gets picked up here as long as a
    # matching --flag was also added above (no per-field boilerplate).
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**config_dict)

    # Same rebuild pattern for InferenceConfig.
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.InferenceConfig)
    }
    # Special-case embed_model_path: coerce to str (argparse gave a plain
    # str already, but this also normalizes falsy/None to "" ) since ""
    # is the sentinel InferenceConfig uses to mean "no embedding backbone,
    # use legacy direct-CNN mode" (see gee.py InferenceConfig docs).
    config_dict["embed_model_path"] = str(args.embed_model_path) if args.embed_model_path else ""
    inference_config = gee.InferenceConfig(**config_dict)

    # Same rebuild pattern for MaskConfig.
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.MaskConfig)
    }
    mask_config = gee.MaskConfig(**config_dict)

    # Log file name is derived from the region file's name so concurrent
    # runs against different regions do not clobber each other's logs.
    logpath = args.logdir / f"gee_{args.region_path.name}.log"
    logger = get_logger(logpath)
    # Record the full parsed CLI args verbatim at the top of the log, so a
    # given log file is self-describing (reproducible run configuration).
    logger.info(f"Inference {datetime.now().isoformat()}: {vars(args)}")

    main(data_config, inference_config, mask_config, args, logger)


