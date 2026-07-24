"""FILE OVERVIEW -- sam2_mask.py: CLI driver for SAM2 mine-scar segmentation.

Context
-------
The main Amazon Mining Watch classifier reports mines as coarse 48-px chip
squares.  Area estimation needs precise outlines, so a fine-tuned SAM2
(Segment Anything 2, facebookresearch/sam2; fine-tuned checkpoint
SAM_model_96_px_final.pth, produced by Georgia Tech students) segments the
actual mine scars inside those detections.

What this script does
---------------------
Thin command-line wrapper around gee.py's SAM2_Masker class:
  1. Parses CLI args that mirror the fields of gee.DataConfig / gee.MaskConfig.
  2. Reads a detections GeoJSON (e.g. cumulative mining polygons).
  3. Redirects mask output into a subfolder named after the detections file.
  4. Calls SAM2_Masker.bulk_mask_polygons, which tiles the polygons, clips
     detections per tile, downloads Sentinel-2 pixels per tile from Google
     Earth Engine, and runs SAM2 in threads.  For each tile, SAM2_Masker:
       - converts detection polygons to a pixel-coordinate box prompt using
         the inverse affine transform of the tile's georeferencing;
       - feeds SAM2 an RGB render of the tile (bands B4,B3,B2,
         reflectance/3000 contrast stretch, cast to uint8);
       - keeps the best-scoring of SAM2's 3 candidate masks;
       - adds a soft spatial prior in logit space (0 inside detection
         polygons, quadratic penalty -(distance/prior_sigma)^2 outside,
         prior_sigma = 12 px by default);
       - bilinearly upsamples SAM2's low-res logits to tile resolution,
         Gaussian-smooths (sigma = 2.5 px by default), then thresholds
         log-odds > 0 to produce the binary mask.
  5. Optionally (--cog) calls sam2_build_cog.main to mosaic the per-tile
     masks into Cloud-Optimized GeoTIFFs (COGs).

Invocation example
------------------
    python sam2_mask.py detections_2023-01-01_2023-12-31.geojson \
        --start_date 2023-01-01 --end_date 2023-12-31 --cog

Inputs on disk
--------------
  * Detections GeoJSON (positional argument; date range in the filename).
  * SAM2 repo checkout, base checkpoint, and fine-tuned weights (CLI paths).

Outputs on disk
---------------
Per-tile EPSG:4326 GeoTIFFs under <mask_dir>/<detections stem>/, named
S2L1C_<tilekey>_<start>_<end>-msk.tif (uint8; 1 = mine, 0 = not mine,
nodata = 2) and S2L1C_<tilekey>_<start>_<end>-logits.tif (float32
log-odds, nodata = NaN).  With --cog: COG mosaics, tile_index.parquet,
and stac_catalog.json under <mask_dir>/<detections stem>/cog_outputs/.

Pipeline position
-----------------
classifier detections (GeoJSON) -> [this script: per-tile SAM2 masks]
    -> sam2_build_cog.py (COG mosaics + index + STAC)
    -> sam2_combine_masks.py (quarterly OR-merge of cumulative mask COGs).
"""

# argparse: builds the CLI whose flags mirror gee.py dataclass fields.
import argparse
# dataclasses.fields: enumerate DataConfig/MaskConfig fields generically,
# so new config fields flow through without editing this file.
from dataclasses import fields
# pathlib.Path: filesystem paths for the detections file and mask dirs.
from pathlib import Path
# re: regex used only for the YYYY-MM-DD date validation helper below.
import re

# geopandas: reads the detections GeoJSON into a GeoDataFrame of polygons.
import geopandas as gpd

# gee: project module holding the DataConfig/MaskConfig dataclasses,
# GEE_Data_Extractor (Sentinel-2 downloads), and SAM2_Masker (segmentation).
import gee  
# sam2_build_cog.main mosaics per-tile masks into COGs; run when --cog set.
from sam2_build_cog import main as run_cogging 


def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged.

    Used as an argparse ``type=`` callback for --start_date / --end_date,
    so malformed dates are rejected at parse time with a usage error.

    Args:
        s (str): candidate date string, e.g. "2023-01-01".

    Returns:
        str: ``s`` unchanged when it matches the 4-2-2 digit shape.

    Raises:
        argparse.ArgumentTypeError: if the string is not shaped like
            YYYY-MM-DD.  Only the shape is checked here (e.g. month 99
            would slip through); Google Earth Engine rejects bad dates
            downstream anyway.

    Pipeline role: guards the imagery date window passed to GEE.
    """
    # Anchored regex: exactly four digits, dash, two digits, dash, two.
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s  # valid shape: hand the string straight back to argparse
    # Invalid: raise the argparse-specific error so the CLI prints a
    # friendly "Not a valid date" message and exits with a usage error.
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def main(args):
    """Run one full SAM2 masking pass described by parsed CLI arguments.

    Purpose:
        Orchestrates a segmentation run end to end: build config objects
        from CLI args, load the detection polygons, download Sentinel-2
        tiles from Google Earth Engine, run SAM2 per tile, and optionally
        mosaic the resulting tile masks into COGs.

    Args:
        args (argparse.Namespace): parsed CLI arguments.  Fields used:
            detections (str): path to detection-polygons GeoJSON.
            start_date / end_date (str): YYYY-MM-DD imagery date range.
            gee.DataConfig fields (tilesize px, pad px, collection,
                clear_threshold 0-1, max_workers, image_cache_dir) and
            gee.MaskConfig fields (prior_sigma px, smoothing_sigma px,
                sam2_repo_path, sam2_checkpoint, finetuned_weights,
                sam2_model_cfg, mask_dir).  Any dataclass field with no
                matching CLI flag silently falls back to its default.
            cog (bool): if True, run the COG mosaicking pipeline after.

    Returns:
        None.

    Side effects (files written):
        * Creates <mask_dir>/<detections stem>/ and fills it with per-tile
          EPSG:4326 GeoTIFFs: *-msk.tif (uint8; 1=mine, 0=not mine,
          nodata=2) and *-logits.tif (float32 log-odds, nodata=NaN).
        * With --cog: COG mosaics, tile_index.parquet (GeoParquet tile
          index), and stac_catalog.json under .../cog_outputs/.

    Pipeline role:
        The driver stage between classifier detections and COG building.
    """

    # Path to the detections GeoJSON; its stem names the output subfolder.
    detections_path = Path(args.detections)
    # Imagery date window (YYYY-MM-DD strings) for the GEE S2 composite.
    start_date, end_date = args.start_date, args.end_date

    # Default DataConfig instance: fallback source for any dataclass field
    # that has no matching attribute on the argparse namespace.
    data_defaults = gee.DataConfig()

    # Generic argparse -> dataclass bridge: for every DataConfig field,
    # take args.<field> when the CLI defined that flag, else the default.
    # Keeps the CLI automatically in sync with gee.py's config schema.
    data_config_dict = {
        f.name: getattr(args, f.name, getattr(data_defaults, f.name))
        for f in fields(gee.DataConfig)
    }
    # Materialize the config controlling tiling + GEE download behavior.
    data_config = gee.DataConfig(**data_config_dict)

    # Same trick for MaskConfig (SAM2 model paths + post-processing knobs).
    mask_defaults = gee.MaskConfig()

    # args.<field> if the CLI flag exists, else the MaskConfig default.
    mask_config_dict = {
        f.name: getattr(args, f.name, getattr(mask_defaults, f.name))
        for f in fields(gee.MaskConfig)
    }
    # Config carrying prior_sigma (px), smoothing_sigma (px), SAM2 repo/
    # checkpoint/fine-tuned-weight paths, Hydra config name, and mask_dir.
    mask_config = gee.MaskConfig(**mask_config_dict)

    # Redirect mask_dir to subfolder per detections file
    # e.g. <mask_dir>/dets_2023-01-01_2023-12-31/ -- keeps runs against
    # different detection sets separated so tile outputs never mix.
    detections_stem = detections_path.stem  # filename minus .geojson suffix
    base_mask_dir = Path(mask_config.mask_dir)  # user-supplied root dir
    run_mask_dir = base_mask_dir / detections_stem  # per-run subfolder
    # Create the directory chain up front; fine if it already exists.
    run_mask_dir.mkdir(parents=True, exist_ok=True)
    # Point the config at the per-run folder so SAM2_Masker writes there.
    mask_config.mask_dir = run_mask_dir

    # Load detection polygons (GeoDataFrame; EPSG:4326 lon/lat geometries).
    dets = gpd.read_file(detections_path)
    # Extractor that tiles geometries and downloads Sentinel-2 pixels from
    # Google Earth Engine over [start_date, end_date], one tile at a time.
    data_extractor = gee.GEE_Data_Extractor(start_date, end_date, data_config)
    # SAM2 wrapper: loads base checkpoint + fine-tuned weights and handles
    # per-tile box prompts, prediction, prior, smoothing, thresholding.
    masker = gee.SAM2_Masker(data_extractor, mask_config)
    # The main event: tile the polygons, clip detections per tile, fetch
    # pixels, run SAM2 threaded, write *-msk.tif / *-logits.tif per tile.
    masker.bulk_mask_polygons(dets)

    # Optional post-step: mosaic thousands of small tile rasters into COGs.
    if args.cog:
        # Tile masks were written into the per-run folder set above.
        input_dir = masker.config.mask_dir
        # COGs and metadata land in a subfolder next to the tile masks.
        output_dir = Path(input_dir) / "cog_outputs"

        # GeoParquet index of per-tile stats (mine_pixels, mine_fraction).
        index_out = output_dir / "tile_index.parquet"
        # STAC catalog JSON describing each output COG (bbox, dates, href).
        stac_out = output_dir / "stac_catalog.json"

        # Delegate to sam2_build_cog.main: groups tiles by UTM zone /
        # 8-degree latitude band / raster type / date range, then runs the
        # gdalbuildvrt -> gdalwarp -> gdal_translate COG chain per group.
        run_cogging(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            index_out=str(index_out),
            stac_out=str(stac_out),
            max_workers=args.max_workers,
        )


if __name__ == "__main__":

    # Build the CLI.  Flag names deliberately match the field names of
    # gee.DataConfig / gee.MaskConfig so main() can map them generically.
    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation on detection polygons"
    )
    # Instantiate defaults once so each flag default tracks gee.py exactly.
    mask_defaults = gee.MaskConfig()

    # ----------------------------
    # Required detections file
    # ----------------------------
    # Positional arg: the polygons SAM2 will refine (classifier output,
    # e.g. cumulative mining detections; date range lives in the name).
    parser.add_argument(
        "detections", type=str,
        help="Path to detection polygons GeoJSON (w/ date range in filename)"
    )
    # Imagery window start (YYYY-MM-DD); shape-checked by valid_date above.
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    # Imagery window end (YYYY-MM-DD, inclusive).
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    
    # ----------------------------
    # MaskConfig args
    # ----------------------------
    # Spatial-prior falloff in pixels: outside the detection polygons the
    # SAM2 logits receive a quadratic penalty -(distance/prior_sigma)^2
    # (0 inside the polygons); default prior_sigma = 12 px in gee.py.
    parser.add_argument("--prior_sigma", type=float,
                        default=mask_defaults.prior_sigma,
                        help="Spatial prior falloff (pixels)")

    # Gaussian sigma (px) applied to the upsampled logits before the
    # log-odds > 0 threshold (default 2.5 px) -- de-speckles mask edges.
    parser.add_argument("--smoothing_sigma", type=float,
                        default=mask_defaults.smoothing_sigma,
                        help="Gaussian smoothing sigma (pixels)")

    # Local checkout of facebookresearch/sam2 (model code + Hydra configs).
    parser.add_argument("--sam2_repo_path", type=str,
                        default=mask_defaults.sam2_repo_path,
                        help="Path to SAM2 repository root")

    # Base (pre-fine-tuning) SAM2 checkpoint file.
    parser.add_argument("--sam2_checkpoint", type=str,
                        default=mask_defaults.sam2_checkpoint,
                        help="Path to SAM2 checkpoint")

    # Fine-tuned weights (SAM_model_96_px_final.pth, by Georgia Tech
    # students) adapting SAM2 to 96-px Sentinel-2 mine-scar imagery.
    parser.add_argument("--finetuned_weights", type=str,
                        default=mask_defaults.finetuned_weights,
                        help="Path to fine-tuned SAM2 weights")

    # Hydra model-architecture config; must match the checkpoint above.
    parser.add_argument("--sam2_model_cfg", type=str,
                        default=mask_defaults.sam2_model_cfg,
                        help="Hydra config name (e.g. configs/sam2.1/"
                             "sam2.1_hiera_s.yaml) or path under sam2_repo/sam2/configs/")

    # Root output dir; main() nests a per-detections-file subfolder inside.
    parser.add_argument("--mask_dir", type=str,
                        default=mask_defaults.mask_dir,
                        help="Directory to write SAM2 GeoTIFF masks")

    # ----------------------------
    # DataConfig args
    # ----------------------------
    # Defaults for the GEE download config (tiling, collection, workers).
    data_defaults = gee.DataConfig()

    # NOTE: default here is 256, overriding gee.DataConfig's usual 576 --
    # smaller tiles better suit the input scale SAM2 was fine-tuned at.
    parser.add_argument("--tilesize", type=int,
                        default=256,
                        help="Tile width in pixels")

    # Extra pixels fetched around each tile so masks stay clean at edges.
    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Padding pixels around tiles")

    # Satellite collection key (e.g. S2L1C = Sentinel-2 Level-1C); valid
    # choices come from gee.DataConfig so the list lives in one place.
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite collection")

    # Cloud-score threshold (0-1) used when compositing cloud-free pixels.
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Cloud-free threshold")

    # Thread count for concurrent GEE tile downloads; also passed on to
    # the COG-building step when --cog is set.
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Parallel GEE requests")

    # Optional on-disk cache of downloaded tiles: reruns skip re-download.
    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional raster cache directory")

    # ----------------------------
    # Cogging flag
    # ----------------------------
    # If present, mosaic the per-tile masks into COGs after segmentation.
    parser.add_argument("--cog", action="store_true",
                        help="Run COG mosaicking pipeline after masking")

    # Parse argv and hand off to the orchestration function above.
    args = parser.parse_args()

    main(args)
