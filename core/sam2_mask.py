
import argparse
from dataclasses import fields
from pathlib import Path
import re

import geopandas as gpd

import gee
from inference_engine import MaskConfig, SAM2_Masker
from sam2_build_cog import main as run_cogging

# >> FILE OVERVIEW: command-line entry point for the SAM2 mask stage (upstream AMW file).
# >> Input: a detections GeoJSON (CNN chip polygons). Output: per-tile SAM2 mask + logits
# >> GeoTIFFs, optionally mosaicked into COGs. Sits after persistence/post-processing and
# >> before the COG/area steps. SAM2 = Meta's "Segment Anything 2" model, fine-tuned here
# >> to trace the exact outline of mine scars inside the coarse 480 m detection chips.
# >> Called by scripts/pipeline.py (cmds_mask_annual / cmds_mask_quarterly print these
# >> runs).

# >> Checks that a command-line date looks like YYYY-MM-DD (format only, not a real date).
# >> Used as the argparse "type" for --start_date / --end_date below.
# >> Returns the string unchanged; raises ArgumentTypeError otherwise.
def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

# >> Main job: build the two configs, run SAM2 on every tile that holds a detection,
# >> and (with --cog) mosaic the per-tile outputs into Cloud-Optimized GeoTIFFs.
# >> Inputs: parsed argparse namespace. Writes into <mask_dir>/<detections file stem>/.
# >> Calls gee.GEE_Data_Extractor (Earth Engine pixels), inference_engine.SAM2_Masker
# >> and sam2_build_cog.main. Returns nothing.
def main(args):

    detections_path = Path(args.detections)
    start_date, end_date = args.start_date, args.end_date

    # >> DataConfig = tile size, padding, satellite collection, cloud threshold, workers.
    # >> Below: take each DataConfig field from the CLI if given, else from the defaults.
    # >> Gotcha: every field that has a CLI flag always has an argparse default, so the
    # >> getattr fallback only matters for fields with no flag at all.
    data_defaults = gee.DataConfig()

    data_config_dict = {
        f.name: getattr(args, f.name, getattr(data_defaults, f.name))
        for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**data_config_dict)

    # >> MaskConfig = SAM2 settings (prior sigma 12 px, smoothing sigma 2.5 px, clamp
    # >> +/-16, model/checkpoint paths, output folder). Same "CLI or default" merge as
    # >> above.
    mask_defaults = MaskConfig()

    mask_config_dict = {
        f.name: getattr(args, f.name, getattr(mask_defaults, f.name))
        for f in fields(MaskConfig)
    }
    mask_config = MaskConfig(**mask_config_dict)

    # >> One output subfolder per detections file, named after the file (e.g. its stem
    # >> carries the date range). pipeline.py later relies on this folder naming.
    # Redirect mask_dir to subfolder per detections file
    detections_stem = detections_path.stem
    base_mask_dir = Path(mask_config.mask_dir)
    run_mask_dir = base_mask_dir / detections_stem
    run_mask_dir.mkdir(parents=True, exist_ok=True)
    # >> Point the config at the subfolder BEFORE SAM2_Masker is built, because the masker
    # >> writes mask_config.txt (provenance of how logits were made) into mask_dir on
    # >> init.
    mask_config.mask_dir = run_mask_dir

    # >> Read the detection polygons (GeoJSON, EPSG:4326 lon/lat) into a GeoDataFrame.
    dets = gpd.read_file(detections_path)
    # >> Earth Engine extractor for this date window. Gotcha: EE filterDate end is
    # >> EXCLUSIVE, so --end_date 2023-12-31 drops 31 December from the composite.
    data_extractor = gee.GEE_Data_Extractor(start_date, end_date, data_config)
    # >> Loads SAM2 weights (GPU if available) and writes mask_config.txt.
    masker = SAM2_Masker(data_extractor, mask_config)
    # >> Real work: tile the detections' footprint, fetch pixels per tile (thread pool of
    # >> max_workers, batches of 500 tiles), run SAM2 serially under a lock, and write
    # >> <collection>_<tilekey>_<start>_<end>-msk.tif (uint8, nodata 2) and -logits.tif
    # >> (float32, NaN nodata), both EPSG:4326, deflate-compressed.
    masker.bulk_mask_polygons(dets)

    # >> Optional: glue the per-tile files into mosaics right away (pipeline.py instead
    # >> runs sam2_build_cog.py as its own "cog" stage).
    if args.cog:
        input_dir = masker.config.mask_dir
        output_dir = Path(input_dir) / "cog_outputs"

        # >> tile_index.parquet = table of every output COG and its footprint.
        # >> stac_catalog.json = STAC (SpatioTemporal Asset Catalog) JSON, a standard
        # >> index for geo datasets.
        index_out = output_dir / "tile_index.parquet"
        stac_out = output_dir / "stac_catalog.json"

        # >> COG = Cloud-Optimized GeoTIFF (internally tiled, readable over HTTP in
        # >> pieces).
        run_cogging(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            index_out=str(index_out),
            stac_out=str(stac_out),
            max_workers=args.max_workers,
            extent_mode=args.extent_mode,
        )


# >> Command-line interface: defines flags, then calls main().
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation on detection polygons"
    )
    mask_defaults = MaskConfig()

    # ----------------------------
    # Required detections file
    # ----------------------------
    # >> Positional argument: the detections GeoJSON to segment.
    parser.add_argument(
        "detections", type=str,
        help="Path to detection polygons GeoJSON (w/ date range in filename)"
    )
    # >> Date window of the satellite composite. Defaults cover 2023 (end exclusive in
    # >> EE).
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    
    # ----------------------------
    # MaskConfig args
    # ----------------------------
    # >> Spatial prior: log-odds penalty -(dist_px/12)^2 outside detected chips, so SAM2
    # >> cannot paint far from where the CNN saw a mine. Units: pixels (~10 m each).
    parser.add_argument("--prior_sigma", type=float,
                        default=mask_defaults.prior_sigma,
                        help="Spatial prior falloff (pixels)")

    # >> Gaussian blur (sigma in pixels) applied to the upsampled logits before "> 0".
    parser.add_argument("--smoothing_sigma", type=float,
                        default=mask_defaults.smoothing_sigma,
                        help="Gaussian smoothing sigma (pixels)")

    # >> Where the SAM2 code and its weights live (default checkpoint
    # >> sam2.1_hiera_small.pt, fine-tuned weights SAM_model_96_px_final.pth inside that
    # >> repo).
    parser.add_argument("--sam2_repo_path", type=str,
                        default=mask_defaults.sam2_repo_path,
                        help="Path to SAM2 repository root")

    parser.add_argument("--sam2_checkpoint", type=str,
                        default=mask_defaults.sam2_checkpoint,
                        help="Path to SAM2 checkpoint")

    parser.add_argument("--finetuned_weights", type=str,
                        default=mask_defaults.finetuned_weights,
                        help="Path to fine-tuned SAM2 weights")

    parser.add_argument("--sam2_model_cfg", type=str,
                        default=mask_defaults.sam2_model_cfg,
                        help="Hydra config name (e.g. configs/sam2.1/"
                             "sam2.1_hiera_s.yaml) or path under sam2_repo/sam2/configs/")

    parser.add_argument("--mask_dir", type=str,
                        default=mask_defaults.mask_dir,
                        help="Directory to write SAM2 GeoTIFF masks")

    # ----------------------------
    # DataConfig args
    # ----------------------------
    data_defaults = gee.DataConfig()

    # >> SAM2 tiles are smaller than CNN tiles: 256 px core + 12 px pad per side = 280 px
    # >> (~2.8 km at 10 m). Hard-coded here, not taken from DataConfig (whose default is
    # >> 552).
    parser.add_argument("--tilesize", type=int,
                        default=256,
                        help="Tile width in pixels")

    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Padding pixels around tiles")

    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite collection")

    # >> Cloud Score+ cutoff (cs_cdf >= 0.6 means "clear enough") applied per scene in EE.
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Cloud-free threshold")

    # >> Number of parallel download threads (Earth Engine requests are network I/O).
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Parallel GEE requests")

    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional raster cache directory")

    # ----------------------------
    # Cogging flag
    # ----------------------------
    parser.add_argument("--cog", action="store_true",
                        help="Run COG mosaicking pipeline after masking")
    # >> 'union' = mosaic only covers tiles present; 'band' = the fixed UTM-zone x
    # >> 8-degree latitude-band box (identical grid every period, but huge and
    # >> memory-hungry).
    parser.add_argument(
        "--extent_mode", choices=("band", "union"), default="union",
        help=(
            "Output extent for per-band mosaics when --cog is set. 'union' "
            "(default) writes the snapped union of the tiles present, on the "
            "global lattice. 'band' writes the fixed UTM-zone x lat-band box "
            "for byte-identical grids across periods, but that is 6.2 Gpx per "
            "band whatever the content -- up to 79,000x the real extent on "
            "sparse bands, and has exhausted memory in gdalwarp."
        ))

    args = parser.parse_args()

    main(args)
