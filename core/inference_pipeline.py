# >> FILE OVERVIEW
# >> Command-line entry point for one inference run: region GeoJSON + date window ->
# >> DLTile tiles -> InferenceEngine.bulk_predict (download, chip, CNN score, keep
# >> chips above --pred_threshold) -> detections GeoJSON under data/outputs/<model>/.
# >> scripts/pipeline.py emits 7 of these commands per period (6 subregions + Andes).
# >> Upstream file (Amazon Mining Watch), byte-identical. Run from inside core/.
import argparse
from dataclasses import fields, MISSING
from datetime import datetime
import logging
import logging.handlers
import re
from pathlib import Path

import geopandas as gpd

# >> Sibling modules in core/ (imported by bare name, so cwd / sys.path must be core/).
import gee
from inference_engine import InferenceConfig, InferenceEngine, MaskConfig
import tile_utils

# >> Check a date string looks like YYYY-MM-DD (format only; 2024-13-45 still passes).
# >> Used as argparse type= for --start_date / --end_date.
def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

# >> Set up logging to a rotating file (max ~1 MB per file, 5 old files kept).
# >> Attaches to the ROOT logger, so every module's log lines go to this file.
# >> Gotcha: only a file handler is added, so nothing is echoed to the console.
def get_logger(logpath: Path, maxBytes=1e6,
               backupCount=5, level=logging.INFO) -> logging.Logger:
    # >> Create the log folder if needed (mkdir -p).
    logpath.parent.mkdir(parents=True, exist_ok=True)
    # >> FILE WRITE: log file rotates at maxBytes into .1 ... .5 backups.
    handler = logging.handlers.RotatingFileHandler(
        logpath, maxBytes=maxBytes, backupCount=backupCount
    )
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# >> Run one inference job: tile the region, score every tile, log a summary.
# >> Inputs: the three config objects built from CLI flags, the parsed args, logger.
# >> Output: none returned; bulk_predict writes the detections GeoJSON as it goes
# >> (rewritten after every batch of 500 tiles).
# >> Called by: the __main__ block below.
def main(data_config: gee.DataConfig,
         inference_config: InferenceConfig,
         mask_config: MaskConfig,
         cli_args: argparse.Namespace,
         logger: logging.Logger):

    # >> FILE READ: region GeoJSON; union_all() merges all features into one geometry.
    region = gpd.read_file(cli_args.region_path).union_all()

    # >> Region -> list of DLTiles (552 px + 12 px pad = 576 px squares at 10 m).
    tiles = tile_utils.create_tiles(
        region, data_config.tilesize, data_config.pad)
    logger.info(f"Created {len(tiles)} tiles")

    # >> Loads the Keras CNN ensemble and creates the EE downloader (ee.Initialize here).
    engine = InferenceEngine(
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
        data_config=data_config,
        config=inference_config,
        mask_config=mask_config,
        logger=logger,
    )
    # >> Main work: producer threads download tiles, one consumer runs the model.
    # >> region_path.stem (e.g. 'Amazon_ACA_1') labels the output GeoJSON.
    preds = engine.bulk_predict(tiles, cli_args.region_path.stem)

    # >> Gotcha: (552/100)^2 = 30.47 per tile is km^2, NOT ha (1 km^2 = 100 ha), so the
    # >> 'ha analyzed' log below is 100x too small. It also ignores the 12 px pad.
    analyzed_area = len(tiles) * (data_config.tilesize / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above "
                f"{inference_config.pred_threshold}")


# >> Collect InferenceConfig's default values (fields that have one) for the CLI.
# >> Used below so CLI defaults always track the dataclass defaults.
def _inference_config_field_defaults():
    """Defaults for :class:`InferenceConfig` fields that have them (excludes ``model_path``)."""
    return {
        f.name: f.default
        for f in fields(InferenceConfig)
        if f.default is not MISSING
    }


# >> CLI section: build defaults from the dataclasses, declare flags, parse, then
# >> rebuild the three config objects from the parsed flags and call main().
if __name__ == '__main__':
    # >> Default DataConfig: tilesize 552, pad 12, S2L1C, clear_threshold 0.6, 8 workers.
    data_defaults = gee.DataConfig()
    inference_field_defaults = _inference_config_field_defaults()
    # >> Default MaskConfig (SAM2 settings); only used if --run_sam2 is given.
    mask_defaults = MaskConfig()

    parser = argparse.ArgumentParser(description="Run bulk ML inference.")

    # DataConfig args
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


    # InferenceConfig args
    # >> Required. pipeline.py passes '--model ...'; argparse accepts that unambiguous
    # >> prefix of --model_path.
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to saved Keras classifier (.h5)",
    )
    # >> Default 0.5 here; pipeline.py passes 0.4 (basin) or 0.2 (Andes).
    parser.add_argument("--pred_threshold", type=float,
                        default=inference_field_defaults["pred_threshold"],
                        help="Prediction threshold for positive chips")
    # >> stride = chip_size // stride_ratio: 48 // 2 = 24 px (chips overlap by half).
    parser.add_argument("--stride_ratio", type=int,
                        default=inference_field_defaults["stride_ratio"],
                        help="Stride is computed as chip_size//stride_ratio")
    # >> Tiles per batch in bulk_predict (default 500); GeoJSON is rewritten per batch.
    parser.add_argument("--max_concurrent_tiles", type=int,
                        default=inference_field_defaults["max_concurrent_tiles"],
                        help="Maximum number of tiles to process concurrently")
    # >> Retry rounds in bulk_predict (default 2; pipeline.py passes 3).
    parser.add_argument("--tries", type=int,
                        default=inference_field_defaults["tries"],
                        help="Number of retries per tile")
    # >> The embedding flags below are for the unused SSL4EO foundation-model paths.
    # >> Production = embedding_strategy 'none' (CNN on raw pixels); pipeline.py never
    # >> overrides it, and the SSL4EO weights are not in the repo.
    parser.add_argument("--embed_model_name", type=str,
                        default=inference_field_defaults["embed_model_name"],
                        help="Embedding model identifier")
    parser.add_argument(
        "--embed_model_path",
        type=str,
        default=None,
        help=(
            "Foundation model weights; if omitted, default path for "
            "embed_model_name=ssl4eo_vit_s16 depends on --embedding_strategy "
            "(cls_only vs cls_patch). Not used when --embedding_strategy none."
        ),
    )
    parser.add_argument("--embed_model_chip_size", type=int,
                        default=inference_field_defaults["embed_model_chip_size"],
                        help="Input size for embedding model")
    parser.add_argument("--embedding_batch_size", type=int,
                        default=inference_field_defaults["embedding_batch_size"],
                        help="Batch size for embedding model inference")
    parser.add_argument("--geo_chip_size", type=int,
                        default=inference_field_defaults["geo_chip_size"],
                        help="Chip size in pixels for cls_only stride cutting, or "
                             "per-window size for cls_patch (typically 224; must match "
                             "embed_model_chip_size when --embedding_strategy cls_patch)")
    parser.add_argument(
        "--embedding_strategy",
        type=str,
        choices=["cls_only", "cls_patch", "none"],
        default=inference_field_defaults["embedding_strategy"],
        help=(
            "cls_only: frozen FM, embed(), legacy *_embeddings.parquet. "
            "cls_patch: ViT intermediate layers, embed_dense(), *_embed_dense_*.parquet. "
            "none: Keras model on pixel chips only (no foundation model)."
        ),
    )
    parser.add_argument("--embeddings_cache_dir", type=str,
                        default=inference_field_defaults["embeddings_cache_dir"],
                        help=("Optional directory to save/reload embeddings. "
                              "Tiles read from cache skip fetching pixels; "
                              "with --run_sam2, inline SAM2 does not run for "
                              "those tiles (use standalone SAM2 masking if needed)."))
    # >> SAM2 = Segment Anything 2, draws mine outlines inside detected chips (optional).
    parser.add_argument("--run_sam2", action="store_true",
                        default=inference_field_defaults["run_sam2"],
                        help="Enable SAM2 segmentation after model predictions")
    parser.add_argument(
        "--inference_output_base",
        type=str,
        default=str(inference_field_defaults["inference_output_base"]),
        help=(
            "Base directory for prediction GeoJSON outputs "
            "(default: <repo>/data/outputs; subfolder per model version)"
        ),
    )
    # MaskConfig args
    # >> SAM2 prior: pixels outside detected chips get -(dist_px/12)^2 added to logits.
    parser.add_argument("--prior_sigma", type=float,
                        default=mask_defaults.prior_sigma,
                        help="Spatial prior falloff (pixels)")
    # >> Gaussian smoothing sigma (px) applied to upsampled SAM2 logits (default 2.5).
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

    # General args
    # >> Relative to cwd (core/). Default = whole basin; pipeline.py passes a subregion.
    parser.add_argument("--region_path", type=Path,
                        default="../data/boundaries/amazon_basin.geojson",
                        help="Path to ROI geojson")
    # >> EE treats end_date as EXCLUSIVE, so the default 2023-12-31 skips Dec 31.
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    # >> Log folder, relative to cwd.
    parser.add_argument("--logdir", type=Path,
                        default=Path("../logs"),
                        help="Directory for log files")

    args = parser.parse_args()

    # >> Rebuild each dataclass from the flags whose names match its fields.
    # >> Gotcha: a field with no CLI flag gets None, not its default. For InferenceConfig
    # >> that is coordinate_precision (default 6) and for MaskConfig logit_clamp
    # >> (default 16). So with --run_sam2 the logits are NOT clamped (`if clamp:` is
    # >> False), and GeoJSON precision is passed as None (driver default, likely full
    # >> precision instead of the intended 6 decimals).
    config_dict = {
        # >> 'bands' has no flag -> None -> DataConfig.__post_init__ fills the 13 S2
        # >> bands.
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**config_dict)

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(InferenceConfig)
    }
    inference_config = InferenceConfig(**config_dict)

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(MaskConfig)
    }
    mask_config = MaskConfig(**config_dict)

    # >> FILE WRITE: logs/gee_<region file name>.log (rotating).
    logpath = args.logdir / f"gee_{args.region_path.name}.log"
    logger = get_logger(logpath)
    # >> Record every CLI argument at the top of the log (provenance of the run).
    logger.info(f"Inference {datetime.now().isoformat()}: {vars(args)}")

    main(data_config, inference_config, mask_config, args, logger)


