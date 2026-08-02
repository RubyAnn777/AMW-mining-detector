# PROVENANCE: FORK-MODIFIED — upstream file modified in this fork; upstream original: earthrise-media/mining-detector
"""Tiles an AOI and exports data from GEE to GeoTiffs for supported collections.

This is a utility function for uses outside the main ML workflow.

Usage: 
python gee_data_pull.py my_aoi.geojson --start_date 2024-01-01 --end_date 2024-01-31 --collection AlphaEarth --tilesize 224

-----------------------------------------------------------------------------
ANNOTATION NOTES (comments/docstrings only; executable code is unchanged)
-----------------------------------------------------------------------------

Where this script fits
    The Amazon Mining Watch (AMW) detector pipeline tiles a region
    (GeoJSON, EPSG:4326) into 576x576-pixel Sentinel-2 tiles (10 m/pixel,
    so ~5.76 km per side) on the Descartes Labs DLTile UTM grid. For each
    tile it builds a cloud-masked median composite in Google Earth Engine
    (GEE): collection COPERNICUS/S2_HARMONIZED is joined with the Cloud
    Score+ quality band "cs_cdf"; pixels with cs_cdf below clear_threshold
    (default 0.6) are masked out, then a per-pixel median is taken over the
    date window. The composite is downloaded via ee.data.computePixels as a
    NumPy array of uint16 digital numbers (reflectance x 10000). In the ML
    workflow, each tile is then cut into 48x48-pixel chips (stride 24 ->
    529 overlapping chips per tile, each covering ~23 ha), normalized
    (/10000, clipped to [0, 1]), bicubic-resized to 224x224, embedded by a
    frozen SSL4EO ViT-S/16 (384-d output), and classified by a Keras
    ensemble of six 384->64->16->1 sigmoid MLPs; chips whose mean sigmoid
    ("confidence") exceeds a threshold are saved as GeoJSON squares.

    THIS SCRIPT SITS OUTSIDE THAT ML WORKFLOW. It reuses the same tiling
    and GEE download machinery (core/gee.py and core/tile_utils.py) but skips
    chipping, embedding, and classification entirely: it simply saves the
    raw composite imagery for each tile to disk as GeoTIFFs. Typical uses:
    debugging the imagery the model actually sees, inspecting cloud-mask
    behavior, and pulling imagery to build figures.

Inputs
    geojson_path (positional): AOI geometries. Any CRS is accepted; the
        file is reprojected to EPSG:4326 on read. Polygon/MultiPolygon
        geometries are tiled on the DLTile UTM grid; Point/MultiPoint
        geometries instead yield one CenteredTile each -- a DLTile-like
        tile centered on the point rather than snapped to the global grid
        (handy for pulling a fixed-size image around a known site).
    --start_date / --end_date: compositing window, YYYY-MM-DD strings
        (validated by valid_date below).
    DataConfig knobs mirrored as CLI flags: --tilesize (pixels per tile
        side), --pad (extra border pixels per side), --collection (e.g.
        S2L1C, S2L2A, S1, AlphaEarth), --clear_threshold (Cloud Score+
        cs_cdf cutoff for Sentinel-2 collections), --max_workers (a
        DataConfig field for concurrent GEE requests; downloads in this
        script happen serially, so it is carried along but not exercised).

Outputs / side effects
    One GeoTIFF per intersecting tile, written into a directory created
    next to the input file and named <aoi_basename><collection>/ (e.g.
    my_aoiS2L1C/ for my_aoi.geojson pulled from S2L1C). For Sentinel-2
    collections the pixel values are stored as uint16 digital numbers
    (reflectance x 10000); rasters are georeferenced in EPSG:4326.

Environment requirements
    This script inserts ../core onto sys.path so it can import gee.py and
    tile_utils.py. NOTE: importing gee immediately calls ee.Initialize(),
    so Google Earth Engine authentication must already be set up and the
    EE_PROJECT environment variable should name your GEE cloud project.
"""

# --- Standard library imports ---
# argparse: builds the CLI in the __main__ block; also supplies
# ArgumentTypeError for the valid_date validator.
import argparse
# dataclasses.fields: used at the bottom to enumerate gee.DataConfig's
# fields so matching CLI flags can be collected into a config dict.
from dataclasses import fields
# pathlib.Path: output-directory handling and the sys.path manipulation.
from pathlib import Path
# re: regex validation of the YYYY-MM-DD date strings.
import re
# sys: needed to prepend the sibling core/ package directory to sys.path.
import sys

# --- Third-party imports ---
# geopandas: reads the AOI GeoJSON into a GeoDataFrame (one row per
# geometry) and handles reprojection to EPSG:4326.
import geopandas as gpd
# tqdm: progress bar over AOI geometries during the download loop.
from tqdm import tqdm

# --- Local imports (from the sibling core/ directory) ---
# This file lives in utils/, but the GEE machinery lives in core/ at the
# repo root. Resolve <repo>/core and put it at the front of sys.path so the
# two modules below can be imported as if they were top-level modules.
# FORK FIX 2026-08-02: upstream still inserts ../gee here — stale since their
# gee/ -> core/ rename (bug present in upstream/main). Point at core/ instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
# gee: provides DataConfig (imagery source + cloud threshold settings) and
# GEE_Data_Extractor (builds the lazy composite; downloads/saves tiles).
# SIDE EFFECT: importing gee runs ee.Initialize() at module import time,
# which requires valid Earth Engine credentials (and uses EE_PROJECT).
import gee
# tiles_for_geometry: returns the tiles covering a geometry. For
# Polygon/MultiPolygon it delegates to the DLTile grid (create_tiles); for
# Point/MultiPoint it returns CenteredTile objects, one per point,
# centered on the point instead of snapped to the DLTile grid.
from tile_utils import tiles_for_geometry

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged.

    Used as an argparse ``type=`` callable for --start_date / --end_date,
    so a malformed date surfaces as an immediate CLI usage error instead
    of a confusing Earth Engine filterDate failure later on.

    Parameters
    ----------
    s : str
        Candidate date string, e.g. ``"2024-01-31"``.

    Returns
    -------
    str
        The same string, unmodified (Earth Engine accepts ISO date
        strings directly, so no conversion to datetime is needed).

    Raises
    ------
    argparse.ArgumentTypeError
        If ``s`` does not match the ``YYYY-MM-DD`` pattern. Note this is
        purely a format check: calendar-impossible dates such as
        ``2024-13-45`` still pass and would only fail inside GEE.
    """
    # Anchored regex: exactly 4 digits, dash, 2 digits, dash, 2 digits.
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # argparse catches this exception type and prints a usage error.
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def main(args):
    """Pull raster data from Earth Engine.

    Tiles every geometry in the input GeoJSON, downloads the (lazily
    defined) composite for each intersecting tile, and writes one GeoTIFF
    per tile.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Relevant attributes:
        - geojson_path (str): path to the AOI GeoJSON file.
        - start_date / end_date (str): ``YYYY-MM-DD`` compositing window.
        - collection (str): collection ID; used here to name the output
          directory (the extractor reads it from args.config).
        - config (gee.DataConfig): assembled in the ``__main__`` block from
          the CLI flags; carries tilesize (px), pad (px), collection,
          bands (resolved automatically per collection), clear_threshold
          (Cloud Score+ cs_cdf cutoff, unitless in [0, 1]), and
          max_workers.

    Returns
    -------
    list
        The tile objects (DLTile or CenteredTile) successfully written to
        disk. Length may be smaller than the number of candidate tiles if
        individual downloads failed.

    Side effects
    ------------
    - Creates the output directory ``<geojson basename><collection>/``.
    - Writes one GeoTIFF per tile into it via ``extractor.save_tile``
      (EPSG:4326; uint16 digital numbers for Sentinel-2 collections).
    - Prints per-tile failures and a final summary to stdout.
    - Issues network requests to Google Earth Engine.
    """
    # Accumulates every tile that was successfully downloaded and saved,
    # both for the summary print and as the return value.
    tiles_written = []

    # Build the extractor. Its constructor immediately assembles the lazy
    # Earth Engine composite for [start_date, end_date): for Sentinel-2
    # collections that is S2 joined with Cloud Score+ (band "cs_cdf"),
    # pixels masked where cs_cdf < clear_threshold, then per-pixel median.
    # No pixels are downloaded yet -- the composite is a server-side recipe.
    extractor = gee.GEE_Data_Extractor(
        args.start_date,
        args.end_date,
        args.config
    )

    # Read the AOI file into a GeoDataFrame (one row per feature) and
    # force EPSG:4326 (lon/lat degrees), the CRS the tiling code expects.
    gdf = gpd.read_file(args.geojson_path).to_crs("EPSG:4326")

    # Output directory sits next to the input file: strip the ".geojson"
    # suffix and append the collection name, e.g. "my_aoi" + "S2L1C" ->
    # my_aoiS2L1C/. Created eagerly so save_tile can write into it.
    outdir = Path(args.geojson_path.split('.geojson')[0] + args.collection)
    outdir.mkdir(parents=True, exist_ok=True)

    # Outer loop: one iteration per AOI feature, with a tqdm progress bar.
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Geometries"):
        geom = row.geometry
        # Defensive skip of degenerate features (empty or self-intersecting
        # geometries would break tiling / intersection tests downstream).
        if geom.is_empty or not geom.is_valid:
            continue
        # Candidate tiles covering this geometry. Polygons -> DLTile UTM
        # grid tiles of (tilesize + 2*pad) pixels per side at 10 m/px;
        # Points -> CenteredTile objects centered on the point itself.
        tiles = tiles_for_geometry(
            geom,
            extractor.config.tilesize,
            extractor.config.pad
        )

        # Inner loop: download and save each tile that truly overlaps the
        # feature (the DLTile cover can include edge tiles that only touch
        # the geometry's bounding box, so re-test intersection here).
        for tile in tiles:
            if not row.geometry.intersects(tile.geometry):
                continue
            try:
                # get_tile_data materializes the composite for this tile
                # via ee.data.computePixels -> NumPy array of shape
                # (H, W, bands); for S2 the values are uint16 digital
                # numbers (reflectance x 10000), cast to float32 in memory.
                pixels = extractor.get_tile_data(tile)
                # save_tile writes the array as a georeferenced GeoTIFF in
                # EPSG:4326 (uint16 on disk for S2 collections), named
                # from the collection, tile key, and date range.
                extractor.save_tile(pixels, tile, outdir)
                tiles_written.append(tile)
            except Exception as e:
                # A single failed tile (network hiccup, GEE quota, etc.)
                # should not abort the whole pull -- log and continue.
                print(f"Tile {tile.key} failed: {e}")

    # Human-readable summary, then return the tile list for callers who
    # import main() programmatically.
    print(f"{len(tiles_written)} tiles written from {len(gdf)} geometries.")
    return tiles_written

if __name__ == "__main__":
    # CLI entry point: define arguments, mirror gee.DataConfig fields as
    # flags, then bundle them back into a DataConfig for main().
    parser = argparse.ArgumentParser(
        description=("Pull raster data from GEE."))

    # Required args
    # Positional path to the AOI file (Polygon/MultiPolygon features are
    # tiled on the DLTile grid; Point/MultiPoint get CenteredTiles).
    parser.add_argument(
        "geojson_path", type=str,
        help="GeoJSON polygons for which to pull raster data.",
    )
    # Compositing window bounds; valid_date rejects non-YYYY-MM-DD input.
    parser.add_argument(
        "--start_date", type=valid_date, required=True,
        help="Start date in YYYY-MM-DD format")
    parser.add_argument(
        "--end_date", type=valid_date, required=True,
        help="End date in YYYY-MM-DD format")

    # DataConfig args
    # Instantiate a default DataConfig purely to source default values for
    # the flags below, so CLI defaults always track the dataclass defaults.
    data_defaults = gee.DataConfig()

    # Tile edge length in pixels (default 576 -> ~5.76 km at 10 m/px).
    parser.add_argument("--tilesize", type=int,
                        default=data_defaults.tilesize,
                        help="Tile width in pixels for requests to GEE")
    # Extra border pixels added on every side of each tile (default 0).
    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Number of pixels to pad each tile")
    # Imagery source; choices come from DataConfig's band mapping keys
    # (S1, S2L1C, S2L1C-12band, S2L2A, AlphaEarth). Determines which bands
    # are requested and how the composite is built.
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite image collection")
    # Cloud Score+ cs_cdf cutoff in [0, 1] (default 0.6): S2 pixels whose
    # clear-sky score falls below this are masked out before the median.
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")
    # Concurrency setting carried inside DataConfig (this script downloads
    # serially, but the field is part of the config contract).
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Maximum concurrent GEE requests")

    args = parser.parse_args()

    # Collect exactly the argparse values whose names match DataConfig
    # fields (getattr default None covers fields without a flag, e.g.
    # "bands", which DataConfig.__post_init__ then resolves per collection).
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }

    # Attach the assembled DataConfig to the namespace so main() receives
    # a single validated config object rather than loose flags.
    args.config = gee.DataConfig(**config_dict)
    main(args)
