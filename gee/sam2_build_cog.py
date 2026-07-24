"""FILE OVERVIEW -- sam2_build_cog.py: mosaic SAM2 tile rasters into COGs.

Build mine scar segmentation COGs from tile GeoTiffs.

What this file does
-------------------
Takes the thousands of small per-tile GeoTIFFs written by sam2_mask.py /
gee.SAM2_Masker (*-msk.tif uint8 masks and *-logits.tif float32 log-odds,
all EPSG:4326) and mosaics them into a handful of Cloud-Optimized GeoTIFFs
(COGs).  A COG is an ordinary GeoTIFF whose internal layout (tiled blocks
plus overview pyramids, metadata up front) lets clients read arbitrary
subsets efficiently via HTTP range requests -- ideal for web maps and
cloud object stores.

Grouping
--------
Tiles are grouped so each output COG covers a manageable, geographically
coherent chunk.  The group key is:
  * UTM zone from tile-center longitude: int((lon + 180) // 6) + 1;
  * 8-degree latitude band (e.g. -8..0) from tile-center latitude;
  * raster type: "mask" vs "logits" (from -msk.tif / -logits.tif suffix);
  * date range parsed from the filename via regex (YYYY-MM-DD_YYYY-MM-DD).

Per-group build (3-step GDAL chain in build_cog)
------------------------------------------------
  1. gdalbuildvrt   -> virtual mosaic: an XML index referencing the source
     tiles; no pixel data is copied at this stage.
  2. gdalwarp       -> materialize the mosaic as a real GeoTIFF, snapping
     all tiles onto one aligned grid; ZSTD-compressed, internally tiled.
  3. gdal_translate -of COG -> rewrite as the final COG with overviews;
     PREDICTOR=2 (horizontal differencing, good for integer masks) or
     PREDICTOR=3 (floating-point predictor, good for float32 logits).

Also produced
-------------
  * A GeoParquet tile index (one row per input tile: bounds, group key,
    plus mine_pixels count and mine_fraction stats for mask tiles).
  * One basin-wide mask COG mosaicking all per-group mask COGs.
  * A STAC catalog (SpatioTemporal Asset Catalog -- a JSON
    FeatureCollection standard describing geospatial assets with
    bbox/datetime/asset href) covering every per-group COG.

How it is invoked
-----------------
Called by sam2_mask.py when --cog is passed, or standalone:
    python sam2_build_cog.py <mask_tile_dir> \
        --output_dir out/ --index_out out/tile_index.parquet \
        --stac_out out/stac_catalog.json --max_workers 8

Inputs on disk:  <input_dir>/*.tif per-tile masks/logits (EPSG:4326).
Outputs on disk (in <output_dir>):
  * mining_<type>_<start>_<end>_utm<z>_lat_<a>_<b>_epsg4326.tif per group;
  * mining_mask_<start>_<end>_epsg4326.tif basin-wide mask mosaic;
  * tile_index.parquet (GeoParquet) and stac_catalog.json.

Pipeline position
-----------------
sam2_mask.py (per-tile masks) -> [this script: COG mosaics + metadata]
    -> sam2_combine_masks.py (OR-merges quarterly mask COGs into the
       cumulative "ever mined" raster).
"""

# argparse: standalone CLI parsing when run directly (see __main__ block).
import argparse
# Thread pool for building independent group COGs concurrently; each task
# mostly blocks on GDAL subprocesses, so threads (not processes) suffice.
from concurrent.futures import ThreadPoolExecutor, as_completed
# json: serializes the STAC catalog FeatureCollection at the end.
import json
# os: makedirs, path joins, and cpu_count for the default worker count.
import os
# math.floor: snaps latitudes onto 8-degree band edges (handles negatives).
import math
# pathlib.Path: globbing *.tif tiles in the input directory.
from pathlib import Path
# re: compiled regex extracting the YYYY-MM-DD_YYYY-MM-DD range from names.
import re
# subprocess: shells out to GDAL CLI tools (gdalbuildvrt/warp/translate).
import subprocess
# tempfile: scratch dir for the intermediate VRT + warped TIFF per COG.
import tempfile

# geopandas: assembles per-tile index rows into a GeoDataFrame and writes
# GeoParquet (geometry column = tile bounding boxes in EPSG:4326).
import geopandas as gpd
# rasterio: opens tiles/COGs to read bounds and mask pixel arrays.
import rasterio
# shapely box(): builds the rectangular footprint geometry for each tile.
from shapely.geometry import box
# tqdm: progress bars for the tile scan and the parallel mosaic builds.
from tqdm import tqdm

# Filename suffixes distinguishing the two per-tile products written by
# gee.SAM2_Masker (see sam2_mask.py for the naming scheme).
MASK_SUFFIX = "-msk.tif"      # uint8 mask: 1=mine, 0=not mine, nodata=2
LOGIT_SUFFIX = "-logits.tif"  # float32 log-odds; NaN marks nodata

# Nodata values kept as strings, ready to splice into GDAL CLI arguments.
MASK_NODATA = "2"     # masks: byte value 2 = no valid data
LOGIT_NODATA = "nan"  # logits: IEEE NaN = no valid data

LAT_BAND_SIZE = 8  # degrees
# Height of each latitude grouping band; 8 deg keeps single mosaics from
# spanning the whole basin north-to-south.

DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")
# Matches the "<start>_<end>" ISO date pair embedded in tile filenames,
# e.g. S2L1C_<tilekey>_2023-01-01_2023-12-31-msk.tif.

def run(cmd):
    """Run an external GDAL command, raising on failure, silencing output.

    Args:
        cmd (list[str]): argv-style command, e.g. ["gdalwarp", src, dst].

    Returns:
        None.

    Side effects:
        Executes the subprocess (which typically writes raster files).
        check=True raises subprocess.CalledProcessError on non-zero exit,
        aborting the pipeline; stdout/stderr go to DEVNULL so the tqdm
        progress bars stay readable during parallel builds.
    """
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def utm_zone_from_lon(lon):
    """Compute the standard UTM zone number for a longitude.

    UTM splits the globe into 60 zones of 6 degrees each starting at
    180 W, so zone = int((lon + 180) // 6) + 1; e.g. lon -75 -> zone 18
    (much of the western Amazon).

    Args:
        lon (float): longitude in decimal degrees, [-180, 180).

    Returns:
        int: UTM zone number in 1..60.  Used purely as a grouping key --
        outputs stay in EPSG:4326; nothing is actually reprojected to UTM.
    """
    return int((lon + 180) // 6) + 1  # shift to 0..360, 6-deg bins, 1-based

def lat_band_from_lat(lat):
    """Snap a latitude to its enclosing 8-degree grouping band.

    floor(lat / 8) * 8 gives the band's southern edge; +8 gives the
    northern edge.  E.g. lat -3.2 -> (-8, 0).  math.floor (not int())
    rounds toward -inf so southern-hemisphere values band correctly.

    Args:
        lat (float): latitude in decimal degrees, [-90, 90].

    Returns:
        tuple[int, int]: (band_start, band_end) in whole degrees, with
        band_start <= lat < band_end.  Second element of the group key.
    """
    # Southern edge of the band containing `lat` (multiple of 8 degrees).
    band_start = math.floor(lat / LAT_BAND_SIZE) * LAT_BAND_SIZE
    # Northern edge is exactly one band above the southern edge.
    band_end = band_start + LAT_BAND_SIZE
    return band_start, band_end

def extract_date_range(filename):
    """Parse the imagery date range embedded in a tile filename.

    Tile names look like S2L1C_<tilekey>_<start>_<end>-msk.tif with dates
    as YYYY-MM-DD; DATE_RANGE_RE grabs the first such underscore-joined
    pair found anywhere in the name.

    Args:
        filename (str): tile file name (basename is enough).

    Returns:
        tuple[str | None, str | None]: (start_date, end_date) strings in
        YYYY-MM-DD form, or (None, None) if no date pair is present --
        such tiles fall into the "nodate" group tag downstream.
    """
    # Search (not match): the date pair sits mid-filename after the key.
    m = DATE_RANGE_RE.search(filename)
    if not m:
        return None, None  # undated tile: caller handles the None pair
    # Two capture groups: (start, end) in filename order.
    start, end = m.groups()
    return start, end

def build_cog(
    input_files,
    output_path,
    raster_type="mask",
    nodata=None,
    resampling=None,
    blocksize=512,
    predictor=None,
):
    """
    Build a Cloud-Optimized GeoTIFF (COG) from input rasters.

    Runs the 3-step GDAL chain: gdalbuildvrt (virtual mosaic -- an XML
    index of the inputs, no pixels copied) -> gdalwarp (materialize onto
    one aligned grid, ZSTD-compressed) -> gdal_translate -of COG (final
    COG with internal tiling + overview pyramids so HTTP range requests
    can fetch subsets cheaply).  Intermediates live in a temporary
    directory that is deleted automatically.

    Args:
        input_files (list[str]): list of input file paths.  Any GDAL
            raster works (per-tile GeoTIFFs, or per-group COGs when
            building the basin-wide mosaic); all share EPSG:4326 here.
        output_path (str): path to output COG (.tif).
        raster_type (str): "mask" (uint8 classes) or "logits" (float32
            log-odds); selects the defaults described below.
        nodata (str|float): nodata value to set; defaults to 2 for masks
            and NaN for logits when not given.
        resampling (str): resampling method for gdalwarp; defaults to
            "near" for masks (never invent class values) and "bilinear"
            for logits (continuous field, interpolation is fine).
        blocksize (int): tile size for COG internal blocks, in pixels.
        predictor (int): predictor for compression; defaults to 2
            (horizontal differencing, suits integers) for masks and 3
            (floating-point predictor) for logits.

    Returns:
        None.

    Side effects:
        Writes the finished COG at output_path.

    Raises:
        ValueError: unknown raster_type.
        subprocess.CalledProcessError: any GDAL step fails (via run()).
    """
    # Set defaults based on raster type
    if raster_type == "mask":
        # Categorical uint8 {0,1,2}: nearest-neighbour keeps values pure.
        resampling = resampling or "near"
        predictor = predictor or 2   # integer horizontal differencing
        nodata = nodata or MASK_NODATA  # byte 2 marks missing data
    elif raster_type == "logits":
        # Continuous float32 log-odds: bilinear resampling is appropriate.
        resampling = resampling or "bilinear"
        predictor = predictor or 3   # floating-point predictor
        nodata = nodata or LOGIT_NODATA  # NaN marks missing data
    else:
        # Guard against typos: only the two known products exist.
        raise ValueError(f"Unknown raster_type: {raster_type}")

    # All intermediates live in a throwaway dir, removed on context exit.
    with tempfile.TemporaryDirectory(prefix="sam2_build_cog_") as tmpdir:
        vrt_path = os.path.join(tmpdir, "tmp.vrt")  # virtual mosaic (XML)
        tmp_tif = os.path.join(tmpdir, "tmp.tif")   # materialized mosaic

        # Step 1: virtual mosaic.  Near-instant -- writes an XML index
        # referencing every input tile; no pixel data is read or copied.
        run(["gdalbuildvrt", vrt_path] + input_files)

        # Warp to align tiles and snap to a consistent grid
        # Step 2: read once through the VRT, write a real GeoTIFF with all
        # tiles resampled onto a single consistent pixel grid.
        run([
            "gdalwarp",
            vrt_path,
            tmp_tif,
            "-r", resampling,              # per-type method chosen above
            "-srcnodata", str(nodata),     # value treated as missing (read)
            "-dstnodata", str(nodata),     # value written where missing
            "-co", "TILED=YES",            # internal tiles (COG prereq)
            "-co", "COMPRESS=ZSTD",        # fast, strong lossless codec
            "-co", "BIGTIFF=IF_SAFER",     # allow >4 GB output if needed
            "-co", "NUM_THREADS=ALL_CPUS"  # parallel compression
        ])

        # Step 3: rewrite as the final COG.  The COG driver adds overview
        # pyramids and orders blocks/metadata for HTTP range-request reads.
        run([
            "gdal_translate",
            tmp_tif,
            output_path,
            "-of", "COG",                     # Cloud-Optimized GeoTIFF
            "-co", "COMPRESS=ZSTD",           # keep ZSTD in the final file
            "-co", f"BLOCKSIZE={blocksize}",  # 512x512 internal tiles
            "-co", f"PREDICTOR={predictor}",  # 2=int diff, 3=float pred.
            "-co", "BIGTIFF=YES",             # basin mosaics can be huge
            "-co", "NUM_THREADS=ALL_CPUS",    # parallel compression
            "-a_nodata", str(nodata)          # stamp nodata in metadata
        ])

def main(input_dir, output_dir, index_out, stac_out, max_workers):
    """Scan tile GeoTIFFs, build grouped COGs, a tile index, and a STAC.

    Purpose:
        The whole mosaicking pipeline: (pass 1) scan every tile, compute
        its group key and index stats; write the GeoParquet tile index;
        (pass 2) build one COG per group in parallel; then one basin-wide
        mask COG; finally a STAC catalog describing the per-group COGs.

    Args:
        input_dir (str): directory of per-tile EPSG:4326 GeoTIFFs
            (*-msk.tif uint8 and/or *-logits.tif float32).
        output_dir (str): destination for COGs + metadata (created).
        index_out (str): output path for the GeoParquet tile index.
        stac_out (str): output path for the STAC catalog JSON.
        max_workers (int): thread count for parallel group COG builds
            (threads block on GDAL subprocesses, so ~CPU count is fine).

    Returns:
        None.

    Side effects (files written):
        * <output_dir>/mining_{mask|logits}_<dates>_utm<z>_lat_<a>_<b>_
          epsg4326.tif -- one COG per group.
        * <output_dir>/mining_mask_<dates>_epsg4326.tif -- basin-wide
          mask mosaic (input to sam2_combine_masks.py).
        * index_out (GeoParquet) and stac_out (JSON FeatureCollection).

    Raises:
        ValueError: if no .tif inputs exist, or the mask groups disagree
            on their imagery date range.

    Pipeline role: stage between per-tile masking and quarterly merging.
    """

    # Ensure the destination exists (harmless when already present).
    os.makedirs(output_dir, exist_ok=True)

    # group key -> list of tile paths that will mosaic into one COG.
    groups = {}
    # One dict per input tile; later becomes the GeoParquet tile index.
    tile_index_rows = []

    # Every GeoTIFF in the mask dir (mask + logits tiles intermingled).
    all_tiles = list(Path(input_dir).glob("*.tif"))
    if not all_tiles:
        # Fail fast: an empty folder usually means a wrong input_dir.
        raise ValueError('No geotiffs found.')

    # Pass 1: read each tile's bounds/pixels, assign it to a group, and
    # record an index row.  Sequential; tiles are small so this is cheap.
    for tif in tqdm(all_tiles, desc="Scanning tiles"):

        # Product type is encoded purely in the filename suffix.
        is_mask = tif.name.endswith(MASK_SUFFIX)
        is_logits = tif.name.endswith(LOGIT_SUFFIX)

        if not (is_mask or is_logits):
            continue  # skip strays (e.g. previously built COGs in place)

        # Normalized label used in group keys, filenames, STAC assets.
        raster_type = "mask" if is_mask else "logits"

        with rasterio.open(tif) as ds:
            # Geographic bounds in EPSG:4326 decimal degrees.
            bounds = ds.bounds
            minx, miny, maxx, maxy = bounds

            # Tile center decides which zone/band group the tile joins.
            center_lon = (minx + maxx) / 2
            center_lat = (miny + maxy) / 2

            # 6-degree UTM zone number (grouping key only; no reproject).
            utm_zone = utm_zone_from_lon(center_lon)
            # 8-degree latitude band edges, e.g. (-8, 0).
            lat_start, lat_end = lat_band_from_lat(center_lat)

            # Imagery date range parsed from filename (may be None pair).
            start_date, end_date = extract_date_range(tif.name)

            # Tiles sharing zone, band, product type, and date range get
            # mosaicked together into one output COG.
            key = (utm_zone, lat_start, lat_end, raster_type,
                   start_date, end_date)
            groups.setdefault(key, []).append(str(tif))

            # Band 1 as a numpy array: uint8 (mask) or float32 (logits),
            # shape (height, width); tiles are ~256 px so this is cheap.
            data = ds.read(1)

            # Mask-only stats; stay None for logits tiles.
            mine_pixels, mine_fraction = None, None
            if is_mask:
                # Count of pixels classified "mine" (mask value 1).
                mine_pixels = int((data == 1).sum())
                # Fraction of the whole tile that is mine (denominator
                # includes nodata pixels -- a coarse coverage metric).
                mine_fraction = mine_pixels / data.size

            # Index row: everything needed to find/query this tile later,
            # with its footprint as a shapely box for spatial queries.
            tile_index_rows.append({
                "filename": tif.name,
                "utm_zone": utm_zone,
                "lat_start": lat_start,
                "lat_end": lat_end,
                "raster_type": raster_type,
                "start_date": start_date,
                "end_date": end_date,
                "mine_pixels": mine_pixels,
                "mine_fraction": mine_fraction,
                "geometry": box(minx, miny, maxx, maxy)
            })

    # Persist the tile index as GeoParquet: columnar, compressed, and
    # geometry-aware -- fast to filter by zone/date/mine_fraction later.
    gdf = gpd.GeoDataFrame(tile_index_rows, crs="EPSG:4326")
    gdf.to_parquet(index_out)

    def build_group(group_key):
        """Build the COG for one (zone, band, type, dates) group.

        Runs inside a worker thread of the pool below.

        Args:
            group_key (tuple): (utm_zone, lat_start, lat_end, raster_type,
                start_date, end_date) -- a key of the `groups` dict;
                closes over `groups` / `output_dir` from main().

        Returns:
            tuple: (cog_path, raster_type, utm_zone, lat_start, lat_end,
            start_date, end_date) -- consumed by the basin-mosaic and
            STAC-catalog steps below.

        Side effects:
            Writes the group's COG into output_dir via build_cog().
        """
        # Unpack the composite grouping key into its six parts.
        utm_zone, lat_start, lat_end, raster_type, start_date, end_date = group_key
        # Human-readable date piece; "nodate" when filenames had no range.
        date_tag = f"{start_date}_{end_date}" if start_date and end_date else "nodate"
        # Full tag names the COG unambiguously, e.g.
        # 2023-01-01_2023-12-31_utm19_lat_-8_0_epsg4326.
        tag = f"{date_tag}_utm{utm_zone}_lat_{lat_start}_{lat_end}_epsg4326"
        cog_path = os.path.join(output_dir, f"mining_{raster_type}_{tag}.tif")
        # Run the VRT -> warp -> COG chain over this group's tile list.
        build_cog(
            input_files=groups[group_key],
            output_path=cog_path,
            raster_type=raster_type
        )
        # Echo the key fields back so the caller can build STAC items.
        return cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date


    # Collected (cog_path, raster_type, zone, lat_start, lat_end,
    # start_date, end_date) tuples, one per finished group.
    results = []

    # Pass 2: build all group COGs concurrently.  Threads spend their time
    # blocked on GDAL subprocesses, so the GIL is not a bottleneck.
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(build_group, key) for key in groups.keys()]

        # Collect in completion order; .result() re-raises worker errors
        # so a failed GDAL step aborts the whole run visibly.
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Building mosaics"):
            results.append(future.result())


    # Build one large mask mosaic across all mask COGs
    # (a single basin-wide raster is what sam2_combine_masks.py consumes).
    mask_cogs = [r[0] for r in results if r[1] == "mask"]

    if mask_cogs:
        # Extract unique date range (assuming all identical)
        # Set comprehension over result tuples: index 5 = start, 6 = end.
        start_dates = {r[5] for r in results if r[1] == "mask"}
        end_dates = {r[6] for r in results if r[1] == "mask"}

        # All mask groups must share one date range to label the mosaic.
        if len(start_dates) == 1 and len(end_dates) == 1:
            start_date = start_dates.pop()
            end_date = end_dates.pop()
        else:
            # Mixed runs in one folder: refuse rather than mislabel dates.
            raise ValueError("Mask groups have inconsistent date ranges.")

        # NOTE: big_vrt / tmp_tif below are vestigial (never used) --
        # build_cog manages its own temp files internally.
        big_vrt = os.path.join(output_dir, "big_mask.vrt")
        tmp_tif = os.path.join(output_dir, "big_mask_tmp.tif")
        # Basin-wide filename, e.g. mining_mask_2023-01-01_2023-12-31_
        # epsg4326.tif -- the raster later OR-merged quarter over quarter.
        big_mask_path = os.path.join(
            output_dir,
            f"mining_mask_{start_date}_{end_date}_epsg4326.tif"
        )
        # Mosaic of mosaics: inputs are the per-group mask COGs themselves.
        build_cog(
            input_files=mask_cogs,
            output_path=big_mask_path,
            raster_type="mask"
        )
 
    # One STAC item (a GeoJSON Feature) per per-group COG.  Note: the
    # basin-wide mosaic above is not itself cataloged here.
    stac_items = []

    for cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date in results:
        # Re-open the finished COG for its authoritative bounds (degrees).
        with rasterio.open(cog_path) as ds:
            bounds = ds.bounds

        # Searchable metadata mirroring the grouping key.
        properties = {
            "utm_zone": utm_zone,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "raster_type": raster_type,
            "start_date": start_date,
            "end_date": end_date
        }

        # STAC temporal extent: full-day UTC timestamps from the range.
        if start_date and end_date:
            properties["start_datetime"] = f"{start_date}T00:00:00Z"
            properties["end_datetime"] = f"{end_date}T23:59:59Z"

        # Minimal STAC item: id = COG filename; geometry = footprint
        # polygon (closed 5-point ring from bottom-left); bbox mirrors
        # it; one asset keyed by raster type with the COG media type.
        stac_items.append({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": os.path.basename(cog_path),
            "properties": properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bounds.left, bounds.bottom],
                    [bounds.right, bounds.bottom],
                    [bounds.right, bounds.top],
                    [bounds.left, bounds.top],
                    [bounds.left, bounds.bottom]
                ]]
            },
            "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "assets": {
                raster_type: {
                    "href": cog_path,
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                }
            }
        })

    # Wrap the items in a FeatureCollection: a flat STAC-style catalog
    # that map clients can load to discover every COG (bbox + datetime).
    stac_catalog = {
        "type": "FeatureCollection",
        "stac_version": "1.0.0",
        "description": "Amazon Mining Watch Scar Masks & Logits",
        "features": stac_items
    }

    # Human-readable JSON (indent=2) so the catalog is easy to inspect.
    with open(stac_out, "w") as f:
        json.dump(stac_catalog, f, indent=2)

    print("All processing complete.")  # completion signal for batch logs

if __name__ == "__main__":

    # Standalone CLI (sam2_mask.py --cog calls main() directly instead).
    parser = argparse.ArgumentParser(description="Build mine scar segmentation COGs from tile GeoTiffs.")
    # Required: folder of per-tile masks/logits produced by SAM2_Masker.
    parser.add_argument("input_dir", help="Directory containing EPSG:4326 DLTile GeoTIFFs")
    # Optional outputs; None means "derive from input_dir" (see below).
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/cog_outputs)")
    parser.add_argument("--index_out", default=None, help="Tile index GeoParquet path")
    parser.add_argument("--stac_out", default=None, help="STAC catalog output JSON")
    # Threads for parallel group builds; defaults to the machine CPU count.
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 4, help="Parallel worker count")

    args = parser.parse_args()

    # Fill unset paths with conventional defaults nested under input_dir,
    # matching the layout sam2_mask.py --cog produces.
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "cog_outputs")
    index_out = args.index_out or os.path.join(output_dir, "tile_index.parquet")
    stac_out = args.stac_out or os.path.join(output_dir, "stac_catalog.json")

    main(input_dir, output_dir, index_out, stac_out, args.max_workers)
