""""Build mine scar segmentation COGs from tile GeoTiffs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm

# >> FILE OVERVIEW: glues thousands of small per-tile SAM2 GeoTIFFs into a few big COGs
# >> (upstream AMW file). Runs after sam2_mask.py (it globs *-msk.tif / *-logits.tif in
# >> one run folder) and feeds areas, sam2_persistence.py and the published maps. Called
# >> as the "cog" stage of scripts/pipeline.py and by sam2_mask.py --cog. Jargon: GeoTIFF
# >> = image file with map coordinates. COG = Cloud-Optimized GeoTIFF: tiled inside +
# >> overview levels, so a web map or script can read just the piece it needs over HTTP.
# >> VRT = GDAL "virtual raster", a small XML file that describes a mosaic of other files
# >> (and here, a Python formula to combine them) without copying pixels. GDAL = the
# >> standard C library/command-line toolkit for rasters; this file shells out to
# >> gdalbuildvrt (make a VRT), gdalwarp (resample onto a grid) and gdal_translate
# >> (convert format, here to COG). Environment knobs below: warp threads = 4;
# >> GDAL_VRT_ENABLE_PYTHON=YES lets GDAL run the Python pixel functions embedded in the
# >> VRTs (off by default for security).
# Required for gdalwarp to evaluate the Python pixel function in the mask
# union VRT (see build_mask_union_vrt). Warping that VRT is effectively
# single-threaded, so cap warp threads rather than leaving them unbounded.
os.environ.setdefault("GDAL_NUM_THREADS", "4")
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")
# GDAL's default block cache is 5% of RAM *per process*, so N cogging workers
# will happily reserve 5N%. Chunked assembly needs very little of it -- the
# 3370-tile group completes inside a hard 4 GB cap -- so pin it low and leave the
# rest of the machine alone.
# >> GDAL block cache limit in MB per process (512 MB) so parallel workers don't hog RAM.
os.environ.setdefault("GDAL_CACHEMAX", "512")

# Groups are built concurrently, and each one's peak is set by GDAL's cache plus
# the final COG translate, not by CPU. os.cpu_count() workers (20 here) is a
# memory decision disguised as a parallelism one.
# >> At most 4 groups are mosaicked at once; the limit is memory, not CPU.
DEFAULT_MAX_WORKERS = 4

# >> Filename endings written by gee.save_tile: masks end "-msk.tif", logits
# >> "-logits.tif".
MASK_SUFFIX = "-msk.tif"
LOGIT_SUFFIX = "-logits.tif"

# >> Nodata values as strings (passed to GDAL flags): mask 2 = not observed; logits NaN.
MASK_NODATA = "2"
LOGIT_NODATA = "nan"

# >> Grouping: tiles are binned by UTM zone (6 degrees of longitude wide) x 8-degree
# >> latitude band, and one COG is built per bin per raster type per date range. Keeps
# >> each mosaic a manageable size. (UTM = Universal Transverse Mercator, 60 north-south
# >> strips.)
LAT_BAND_SIZE = 8  # degrees
UTM_ZONE_SIZE = 6  # degrees

# --- Fixed output grid -------------------------------------------------------
# Every published raster sits on one global lattice: pixel edges fall on integer
# multiples of GRID_RES degrees from (0, 0). This makes pixel (i, j) the same
# ground location in every period and every UTM/lat band, so temporal rules need
# no regridding and adjacent bands mosaic without resampling.
#
# GRID_RES is finer than every observed source resolution (masks span
# 9.0404e-05 .. 9.0952e-05 deg), so regridding never decimates. See
# docs/design/persistence-planning.md, "Fixing the raster grid".
# >> The 0.00009-degree lattice: every output pixel is 0.00009 deg square (~10 m N-S; less
# >> E-W away from the equator) with edges on exact multiples from (0, 0). So pixel (i, j)
# >> is the same ground spot in every year, which lets persistence compare years pixel by
# >> pixel.
GRID_RES = 0.00009  # degrees/pixel, ~10.02 m in latitude

# Tiles are assigned to a band by their center, so a tile may overhang the band
# boundary by up to half its width (~0.026 deg for a 576 px tile). This margin
# is the fixed allowance; overruns beyond it expand the extent and warn rather
# than clip.
# >> Extra 0.08 deg (~9 km) around each band box, because tiles near a band edge overhang
# >> it.
GRID_MARGIN_DEG = 0.08

# Mosaics are assembled one chunk at a time. GDAL asks a VRTDerivedRasterBand
# for full-width strips, so the union pixel function allocates
# n_sources x raster_width x strip_height -- and with one VRT spanning a whole
# band, both factors grow with the archive. Measured on 2018 utm21 lat[0,8]:
# 3370 sources x 260 x 66885 = 58 GB, which no amount of RAM makes safe because
# it grows every year. Chunking caps the width at CHUNK_PX and the source count
# at the tiles overlapping one chunk (<= 79 across all 2018 groups), i.e. ~42 MB.
#
# 2048 rather than larger: the worst-case chunk holds 79 tiles at 2048 px but 180
# at 4096, and memory is the product. Rather than smaller: chunk count sets the
# subprocess overhead, which dominates once the stacking cost is gone.
# >> Chunking: each mosaic is built in 2048 x 2048 px pieces (~20 km square) and then
# >> stitched, because GDAL would otherwise allocate (n tiles x width x strip) and run out
# >> of memory (58 GB measured). Only tiles touching a chunk are loaded for it.
CHUNK_PX = 2048

# Guard against float noise when snapping a value that is already on-lattice.
# >> Tiny tolerance so a value already on the lattice does not snap one pixel over.
_SNAP_EPS = 1e-9

# >> Finds the "YYYY-MM-DD_YYYY-MM-DD" date range inside tile filenames.
DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")


# >> Rounds a coordinate DOWN to the nearest lattice line (multiple of 0.00009 deg).
# >> Input/Output: float degrees. Used by snap_extent / resolve_extent.
def snap_floor(value, resolution=GRID_RES):
    """Largest lattice coordinate <= value."""
    return math.floor(value / resolution + _SNAP_EPS) * resolution


# >> Rounds a coordinate UP to the nearest lattice line (multiple of 0.00009 deg).
def snap_ceil(value, resolution=GRID_RES):
    """Smallest lattice coordinate >= value."""
    return math.ceil(value / resolution - _SNAP_EPS) * resolution


# >> Grows a bounding box (minx, miny, maxx, maxy in degrees) outward onto the lattice,
# >> so it fully contains the original box. Used by band_extent and resolve_extent.
def snap_extent(bounds, resolution=GRID_RES):
    """Snap (minx, miny, maxx, maxy) outward onto the lattice."""
    minx, miny, maxx, maxy = bounds
    return (
        snap_floor(minx, resolution),
        snap_floor(miny, resolution),
        snap_ceil(maxx, resolution),
        snap_ceil(maxy, resolution),
    )


# >> Fixed box for one UTM zone x lat band (plus 0.08 deg margin), snapped to the lattice.
# >> Only used when extent_mode="band": same box every period, so grids are
# >> byte-identical. Output: (minx, miny, maxx, maxy) in EPSG:4326 degrees.
def band_extent(utm_zone, lat_start, lat_end, resolution=GRID_RES,
                margin=GRID_MARGIN_DEG):
    """Fixed lattice-aligned extent for a UTM zone x lat band group.

    Derived from the band definition rather than from the tiles present, so the
    extent is identical for every period.
    """
    # >> Western edge of UTM zone n in longitude: zone 1 starts at -180, each zone is 6
    # >> deg.
    lon_start = -180 + UTM_ZONE_SIZE * (utm_zone - 1)
    return snap_extent(
        (lon_start - margin, lat_start - margin,
         lon_start + UTM_ZONE_SIZE + margin, lat_end + margin),
        resolution,
    )


# >> Bounding box that covers all the given raster files (opens each with rasterio).
# >> Input: list of paths. Output: (minx, miny, maxx, maxy) in degrees. Fails on empty
# >> list.
def union_bounds(input_files):
    """Union of the bounds of input_files.

    Raises on an empty list rather than returning infinities, which would
    otherwise surface much later as an incomprehensible OverflowError from
    math.floor inside snap_extent.
    """
    input_files = list(input_files)
    if not input_files:
        raise ValueError("union_bounds: no input rasters")
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for path in input_files:
        # >> Opens each file just to read its header bounds (no pixel data read).
        with rasterio.open(path) as ds:
            b = ds.bounds
        minx, miny = min(minx, b.left), min(miny, b.bottom)
        maxx, maxy = max(maxx, b.right), max(maxy, b.top)
    return (minx, miny, maxx, maxy)


# >> Decides the final output box: the snapped union of the inputs (union mode), or the
# >> fixed band box widened if any tile sticks out (band mode). Always on the lattice.
# >> Called by build_cog. Prints a WARNING when it has to grow the fixed box.
def resolve_extent(input_files, extent, resolution=GRID_RES, label=""):
    """Return a lattice-aligned extent that is guaranteed to contain the inputs.

    Expands (and warns) rather than clipping if the inputs overrun ``extent``:
    the result stays on the lattice either way, so temporal alignment holds even
    when the fixed extent has to grow.
    """
    src = union_bounds(input_files)
    if extent is None:
        return snap_extent(src, resolution)

    grown = (
        min(extent[0], snap_floor(src[0], resolution)),
        min(extent[1], snap_floor(src[1], resolution)),
        max(extent[2], snap_ceil(src[2], resolution)),
        max(extent[3], snap_ceil(src[3], resolution)),
    )
    if grown != tuple(extent):
        print(
            f"WARNING: {label or 'group'} tiles overrun the fixed extent "
            f"{tuple(round(v, 6) for v in extent)} -> "
            f"{tuple(round(v, 6) for v in grown)}. Data is preserved and the "
            f"grid stays aligned, but consider raising GRID_MARGIN_DEG."
        )
    return grown


# >> Cuts the output box into non-overlapping 2048 x 2048 px squares (the chunks). Yields
# >> (minx, miny, maxx, maxy) in degrees. Works in integer pixel indices so seams land
# >> exactly on the lattice. Order: outer loop west->east (i), inner south->north (j).
def chunk_extents(extent, chunk_px=CHUNK_PX, resolution=GRID_RES):
    """Disjoint lattice-aligned boxes tiling ``extent``.

    Edges are derived in integer lattice indices rather than by repeated float
    addition, which would drift and put chunk seams a pixel off the lattice --
    the one thing that would make chunked assembly differ from a single pass.
    """
    i0, j0 = round(extent[0] / resolution), round(extent[1] / resolution)
    i1, j1 = round(extent[2] / resolution), round(extent[3] / resolution)
    # >> i = column index (longitude), j = row index (latitude), stepping by chunk_px.
    for i in range(i0, i1, chunk_px):
        for j in range(j0, j1, chunk_px):
            yield (i * resolution, j * resolution,
                   min(i + chunk_px, i1) * resolution,
                   min(j + chunk_px, j1) * resolution)


# >> Builds one small GeoTIFF per chunk that any tile touches, with overlaps resolved
# >> (mask OR / logit max). Returns list of chunk .tif paths inside tmpdir.
# >> Uses shapely STRtree (spatial index) to find which tiles overlap each chunk quickly.
# >> Called by build_cog; its outputs are stitched there.
def build_chunk_rasters(input_files, extent, raster_type, resampling, nodata,
                        resolution=GRID_RES, chunk_px=CHUNK_PX, tmpdir=None):
    """Warp ``input_files`` into one raster per non-empty chunk of ``extent``.

    Only the tiles intersecting a chunk are fed to that chunk's union VRT, and
    chunks no tile touches are skipped entirely -- for the 2018 groups that is
    90% of them. Skipping is not a shortcut: the assembling VRT leaves unwritten
    ground at its own nodata, which is the same "unobserved, not zero" result a
    full pass produces.
    """
    # >> Footprint rectangle of each tile, from the file header.
    geometries = []
    for path in input_files:
        with rasterio.open(path) as ds:
            geometries.append(box(*ds.bounds))
    # >> STRtree = spatial index; tree.query(box) returns indices of tiles touching the
    # >> box.
    tree = STRtree(geometries)

    chunk_paths = []
    # >> Loop over chunks in chunk_extents order; this defines chunk_0, chunk_1, ...
    # >> numbering.
    for n, chunk in enumerate(chunk_extents(extent, chunk_px, resolution)):
        members = [input_files[k] for k in tree.query(box(*chunk))]
        # >> Empty chunk (no tile): skipped; the stitched mosaic reads it as nodata.
        if not members:
            continue
        # Each chunk gets its own directory: build_mask_union_vrt writes a
        # fixed-name stack.vrt beside its output and the derived band refers to
        # it relatively, so chunks sharing a directory would overwrite it.
        chunk_dir = Path(tmpdir) / f"chunk_{n}"
        chunk_dir.mkdir()
        vrt_path = chunk_dir / "tmp.vrt"
        chunk_tif = chunk_dir / "chunk.tif"

        # >> Masks: VRT with the "mask_or" pixel function. Logits: VRT with "logit_max".
        if raster_type == "mask":
            build_mask_union_vrt(
                members, vrt_path, nodata_val=int(nodata),
                extent=chunk, resolution=resolution, resampling=resampling)
        else:
            build_logit_max_vrt(
                members, vrt_path,
                extent=chunk, resolution=resolution, resampling=resampling)

        # >> gdalwarp materialises the VRT onto the lattice for this chunk: -r resampling
        # >> (near for masks, bilinear for logits); -tr pixel size x y in degrees
        # >> (0.00009); -te target extent minx miny maxx maxy; -srcnodata/-dstnodata keep
        # >> nodata as nodata; -co creation options: TILED=YES internal tiles,
        # >> COMPRESS=ZSTD (fast modern lossless compression), BIGTIFF=IF_SAFER (use
        # >> 64-bit offsets if file might pass 4 GB). Writes chunk_<n>/chunk.tif.
        run([
            "gdalwarp",
            vrt_path,
            chunk_tif,
            "-r", resampling,
            "-tr", repr(resolution), repr(resolution),
            "-te", *[repr(v) for v in chunk],
            "-srcnodata", str(nodata),
            "-dstnodata", str(nodata),
            "-co", "TILED=YES",
            "-co", "COMPRESS=ZSTD",
            "-co", "BIGTIFF=IF_SAFER",
        ])
        chunk_paths.append(chunk_tif)

    return chunk_paths


# >> Runs a shell command (list of args) with subprocess; hides GDAL chatter on success,
# >> raises CalledProcessError with GDAL's stderr + the command on failure.
def run(cmd):
    """Run a GDAL command, surfacing its stderr if it fails.

    GDAL is chatty on success and the noise is not worth keeping, but silencing
    stderr unconditionally makes a failure look like nothing happened at all --
    the traceback names the exit code and no reason. Capture, discard on
    success, re-raise with the message on failure.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd,
            output=proc.stdout,
            stderr=f"{proc.stderr.strip()}\n  command: {' '.join(map(str, cmd))}")

# >> Builds the gdalbuildvrt flags that put a VRT exactly on the lattice.
# >> -resolution user + -tr = use our pixel size; -te = our extent; -r = resampling;
# >> -srcnodata / -vrtnodata = uncovered ground reads as nodata, not 0.
# >> Gotcha: no -tap flag is used anywhere here; alignment comes from snapping -te to the
# >> lattice by hand (gdal's -tap "target aligned pixels" would do the same from -tr).
def _buildvrt_grid_args(extent, resolution, resampling, nodata):
    """gdalbuildvrt arguments pinning the output to the fixed lattice.

    Pinning the VRT (rather than only gdalwarp) means each source is placed on
    the final grid once, instead of being resampled twice.
    """
    # gdalwarp spells nearest "near"; gdalbuildvrt wants "nearest".
    # >> Name translation: gdalwarp says "near", gdalbuildvrt says "nearest".
    resampling = "nearest" if resampling == "near" else resampling
    args = [
        "-resolution", "user",
        "-tr", repr(resolution), repr(resolution),
        "-te", *[repr(v) for v in extent],
        "-r", resampling,
    ]
    # Regions of the fixed extent not covered by a source must read as nodata,
    # not as 0 -- otherwise the mask OR reduces unobserved ground to
    # "observed, not mining".
    args += ["-srcnodata", str(nodata), "-vrtnodata", str(nodata)]
    return args


# >> Makes a VRT whose single band = pixel-wise OR of N mask tiles (1 if any tile says
# >> mine, 2 if all tiles are nodata, else 0). Writes stack.vrt (the N tiles as N bands)
# >> and derived_vrt_path (the formula over it). Needed because plain GDAL mosaics are
# >> "last file wins", which would erase a mine seen by an earlier overlapping tile.
def build_mask_union_vrt(input_files, derived_vrt_path, nodata_val,
                         extent=None, resolution=GRID_RES, resampling="nearest"):
    """Build a VRT computing the pixel-wise union (logical OR) of N mask rasters.

    Plain gdalbuildvrt mosaicking is "last file wins" for overlapping pixels,
    which drops a 1 from an earlier tile if a later tile has 0 in the same
    spot. This instead stacks all inputs as separate bands and reduces them
    with a nodata-aware OR: output is 1 if any input is 1, nodata only if
    every input is nodata, and 0 otherwise.
    """
    # >> stack.vrt sits next to the derived VRT; that's why each chunk needs its own
    # >> folder.
    stack_vrt = Path(derived_vrt_path).parent / "stack.vrt"

    grid_args = (
        _buildvrt_grid_args(extent, resolution, resampling, nodata_val)
        if extent is not None else []
    )
    # >> gdalbuildvrt -separate: each input becomes its own band (no merging yet).
    run(["gdalbuildvrt", "-separate"] + grid_args + [stack_vrt] + input_files)

    # >> Parse the VRT XML, drop its N plain bands, add one derived band that reads them
    # >> all.
    tree = ET.parse(stack_vrt)
    root = tree.getroot()

    for band in root.findall("VRTRasterBand"):
        root.remove(band)

    # >> VRTDerivedRasterBand = a band computed by a pixel function; Byte = uint8.
    derived = ET.SubElement(root, "VRTRasterBand", {
        "dataType": "Byte",
        "band": "1",
        "subClass": "VRTDerivedRasterBand",
    })
    ET.SubElement(derived, "NoDataValue").text = str(nodata_val)
    ET.SubElement(derived, "PixelFunctionType").text = "mask_or"
    ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
    # >> The Python source below is stored inside the VRT XML and executed by GDAL for
    # >> each block: in_ar = list of N arrays (one per tile), out_ar = output block to
    # >> fill. Same logic as sam2_logits.mosaic_masks. The nodata value is pasted into the
    # >> code via the f-string.
    code = ET.SubElement(derived, "PixelFunctionCode")
    code.text = f"""
import numpy as np
def mask_or(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    stack = np.stack(in_ar)
    is_nodata = stack == {nodata_val}
    out_ar[:] = np.where(
        is_nodata.all(axis=0),
        {nodata_val},
        np.where((stack == 1).any(axis=0), 1, 0),
    )
"""
    # >> Point the derived band at every band (1..N) of stack.vrt, by relative filename.
    stack_basename = Path(stack_vrt).name
    for band_index in range(1, len(input_files) + 1):
        src = ET.SubElement(derived, "SimpleSource")
        ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
        ET.SubElement(src, "SourceBand").text = str(band_index)

    # >> Write the derived VRT XML to disk.
    tree.write(derived_vrt_path)

# >> Makes a VRT whose single band = pixel-wise max of N logit tiles, ignoring NaN (NaN
# >> only if every tile is NaN). Float32. Same stack.vrt + derived-band trick as the mask
# >> OR. max(logits) > t is the same as "any tile's mask is 1", so the two mosaics agree.
def build_logit_max_vrt(input_files, derived_vrt_path, extent=None,
                        resolution=GRID_RES, resampling="bilinear"):
    """Build a VRT computing the pixel-wise max of N logit rasters.

    Plain gdalbuildvrt mosaicking is "last file wins" for overlapping pixels,
    so a later tile's weaker logit would silently overwrite an earlier
    tile's stronger one. This instead stacks all inputs as separate bands
    and reduces them with a NaN-aware max: output is the largest non-NaN
    logit at each pixel, NaN only where every input is NaN. (Max is the
    logit-space equivalent of the mask OR: with a shared threshold,
    max(logits) > t iff any per-tile mask would be 1.)
    """
    stack_vrt = Path(derived_vrt_path).parent / "stack.vrt"

    grid_args = (
        _buildvrt_grid_args(extent, resolution, resampling, LOGIT_NODATA)
        if extent is not None else []
    )
    run(["gdalbuildvrt", "-separate"] + grid_args + [stack_vrt] + input_files)

    tree = ET.parse(stack_vrt)
    root = tree.getroot()

    for band in root.findall("VRTRasterBand"):
        root.remove(band)

    derived = ET.SubElement(root, "VRTRasterBand", {
        "dataType": "Float32",
        "band": "1",
        "subClass": "VRTDerivedRasterBand",
    })
    # >> Float32 output with NaN nodata.
    ET.SubElement(derived, "NoDataValue").text = "nan"
    ET.SubElement(derived, "PixelFunctionType").text = "logit_max"
    ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
    # >> Embedded Python pixel function (run by GDAL): NaN -> -inf, max over tiles, -inf
    # >> -> NaN.
    code = ET.SubElement(derived, "PixelFunctionCode")
    code.text = """
import numpy as np
def logit_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    stack = np.stack(in_ar).astype(np.float32)
    filled = np.where(np.isnan(stack), -np.inf, stack)
    reduced = filled.max(axis=0)
    out_ar[:] = np.where(np.isneginf(reduced), np.nan, reduced)
"""
    stack_basename = Path(stack_vrt).name
    for band_index in range(1, len(input_files) + 1):
        src = ET.SubElement(derived, "SimpleSource")
        ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
        ET.SubElement(src, "SourceBand").text = str(band_index)

    tree.write(derived_vrt_path)

# >> Which UTM zone (1..60) a longitude falls in: 6-degree strips starting at -180.
# >> Gotcha: lon exactly 180 gives zone 61 (irrelevant for the Amazon).
def utm_zone_from_lon(lon):
    return int((lon + 180) // 6) + 1

# >> Which 8-degree latitude band a latitude falls in, e.g. -3.2 -> (-8, 0).
def lat_band_from_lat(lat):
    band_start = math.floor(lat / LAT_BAND_SIZE) * LAT_BAND_SIZE
    band_end = band_start + LAT_BAND_SIZE
    return band_start, band_end

# >> Pulls (start_date, end_date) strings out of a tile filename, or (None, None).
def extract_date_range(filename):
    m = DATE_RANGE_RE.search(filename)
    if not m:
        return None, None
    start, end = m.groups()
    return start, end

# >> Makes ONE COG from a list of rasters: pick defaults, choose extent, build chunks,
# >> stitch them with gdalbuildvrt, convert to COG with gdal_translate.
# >> Masks: nearest-neighbour resampling, PREDICTOR=2, nodata 2. Logits: bilinear,
# >> PREDICTOR=3, nodata NaN. Called by main.build_group and for the "big" mosaics.
def build_cog(
    input_files,
    output_path,
    raster_type="mask",
    nodata=None,
    resampling=None,
    blocksize=512,
    predictor=None,
    extent=None,
    resolution=GRID_RES,
    chunk_px=CHUNK_PX,
):
    """
    Build a Cloud-Optimized GeoTIFF (COG) from input rasters.

    Args:
        input_files (list[str]): list of input file paths
        output_path (str): path to output COG
        raster_type (str): "mask" or "logits"
        nodata (str|float): nodata value to set
        resampling (str): resampling method for gdalwarp
        blocksize (int): tile size for COG
        predictor (int): predictor for compression
        extent (tuple|None): fixed (minx, miny, maxx, maxy). Snapped onto the
            lattice and widened if the inputs overrun it. If None, the snapped
            union of the inputs is used -- still lattice-aligned, but the extent
            then varies with the tiles present.
        resolution (float): degrees/pixel; the lattice step. Defaults to
            GRID_RES and should not normally be overridden.
        chunk_px (int): edge length of the assembly chunks, in pixels. Bounds
            peak memory; see CHUNK_PX.
    """
    # Set defaults based on raster type
    # >> PREDICTOR = pre-compression trick: 2 = store differences between neighbouring
    # >> integers (good for 0/1 masks), 3 = same idea for floating-point (logits). Makes
    # >> ZSTD smaller.
    if raster_type == "mask":
        resampling = resampling or "near"
        predictor = predictor or 2
        nodata = nodata or MASK_NODATA
    elif raster_type == "logits":
        resampling = resampling or "bilinear"
        predictor = predictor or 3
        nodata = nodata or LOGIT_NODATA
    else:
        raise ValueError(f"Unknown raster_type: {raster_type}")

    # >> Final output box on the lattice (union of tiles, or the fixed band box).
    extent = resolve_extent(
        input_files, extent, resolution, label=Path(output_path).name)

    # >> Temporary folder for chunks; deleted automatically when the "with" block ends.
    with tempfile.TemporaryDirectory(prefix="sam2_build_cog_") as tmpdir:
        # >> Step 1: per-chunk rasters with overlaps already resolved.
        chunk_paths = build_chunk_rasters(
            input_files, extent, raster_type, resampling, nodata,
            resolution=resolution, chunk_px=chunk_px, tmpdir=tmpdir)
        if not chunk_paths:
            raise ValueError(
                f"no chunk of {extent} is covered by any of the "
                f"{len(input_files)} input rasters")

        # The chunks are disjoint and already on the lattice, so this mosaic
        # never has to resolve a conflict: plain last-wins is exact here, and
        # cheap because GDAL reads only the sources a block intersects. The
        # expensive union already happened, once, inside each chunk.
        mosaic_vrt = Path(tmpdir) / "mosaic.vrt"
        # >> Step 2: plain VRT mosaic of the disjoint chunks over the full extent.
        run(["gdalbuildvrt"]
            + _buildvrt_grid_args(extent, resolution, resampling, nodata)
            + [mosaic_vrt] + chunk_paths)

        # >> Step 3: gdal_translate -of COG writes the final Cloud-Optimized GeoTIFF:
        # >> COMPRESS=ZSTD; BLOCKSIZE=512 (internal tile 512 x 512 px); PREDICTOR 2 or 3
        # >> (see above); BIGTIFF=YES (64-bit offsets, needed above 4 GB);
        # >> NUM_THREADS=ALL_CPUS for compression; -a_nodata stamps the nodata value into
        # >> the file header.
        run([
            "gdal_translate",
            mosaic_vrt,
            output_path,
            "-of", "COG",
            "-co", "COMPRESS=ZSTD",
            "-co", f"BLOCKSIZE={blocksize}",
            "-co", f"PREDICTOR={predictor}",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-a_nodata", str(nodata)
        ])

# >> Entry point: scan a folder of per-tile outputs, write tile_index.parquet, build one
# >> COG per (UTM zone, lat band, raster type, date range), then one all-region mosaic per
# >> type, then a STAC catalog JSON listing every COG. Inputs: input_dir with *-msk.tif /
# >> *-logits.tif; raster_types default ("mask",). Gotcha: sam2_mask.py --cog calls this
# >> without raster_types, so it never builds logits.
def main(input_dir, output_dir, index_out, stac_out, max_workers,
         extent_mode="union", raster_types=("mask",)):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    groups = {}
    tile_index_rows = []

    # >> Every .tif in the folder (not recursive); non-mask/logit files are skipped below.
    all_tiles = list(Path(input_dir).glob("*.tif"))
    if not all_tiles:
        raise ValueError('No geotiffs found.')

    # >> Loop over tiles in glob order: work out each tile's group and its index row.
    for tif in tqdm(all_tiles, desc="Scanning tiles"):

        is_mask = tif.name.endswith(MASK_SUFFIX)
        is_logits = tif.name.endswith(LOGIT_SUFFIX)

        if not (is_mask or is_logits):
            continue

        raster_type = "mask" if is_mask else "logits"
        if raster_type not in raster_types:
            continue

        with rasterio.open(tif) as ds:
            bounds = ds.bounds
            minx, miny, maxx, maxy = bounds

            center_lon = (minx + maxx) / 2
            center_lat = (miny + maxy) / 2

            # >> Group by the tile's CENTRE point, so a tile belongs to exactly one zone x
            # >> band.
            utm_zone = utm_zone_from_lon(center_lon)
            lat_start, lat_end = lat_band_from_lat(center_lat)

            start_date, end_date = extract_date_range(tif.name)

            # >> Group key = (utm_zone, lat_start, lat_end, raster_type, start_date,
            # >> end_date).
            key = (utm_zone, lat_start, lat_end, raster_type,
                   start_date, end_date)
            groups.setdefault(key, []).append(str(tif))

            # >> Reads the whole first band just to count mine pixels (also done for
            # >> logits, unused).
            data = ds.read(1)

            mine_pixels, mine_fraction = None, None
            if is_mask:
                # >> Mine pixel count (value 1) and its share of all pixels in the tile
                # >> (nodata counted in the denominator).
                mine_pixels = int((data == 1).sum())
                mine_fraction = mine_pixels / data.size

            # >> One row per tile for tile_index.parquet.
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

    # >> tile_index.parquet = GeoParquet (columnar table + geometry) with one row per
    # >> tile: filename, zone, band, type, dates, mine pixel count/fraction, footprint box
    # >> (EPSG:4326).
    gdf = gpd.GeoDataFrame(tile_index_rows, crs="EPSG:4326")
    gdf.to_parquet(index_out)

    # >> Builds the COG for one group. Output name:
    # >> mining_<type>_<start>_<end>_utm<zone>_lat_<lo>_<hi>_epsg4326.tif in output_dir.
    # >> Returns a tuple later used for the STAC catalog.
    def build_group(group_key):
        utm_zone, lat_start, lat_end, raster_type, start_date, end_date = group_key
        date_tag = f"{start_date}_{end_date}" if start_date and end_date else "nodate"
        tag = f"{date_tag}_utm{utm_zone}_lat_{lat_start}_{lat_end}_epsg4326"
        cog_path = Path(output_dir) / f"mining_{raster_type}_{tag}.tif"
        build_cog(
            input_files=groups[group_key],
            output_path=cog_path,
            raster_type=raster_type,
            extent=(band_extent(utm_zone, lat_start, lat_end)
                    if extent_mode == "band" else None),
        )
        return cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date


    results = []

    # >> Thread pool (max_workers, default 4): groups built in parallel. Threads are fine
    # >> because the heavy work happens in GDAL subprocesses.
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(build_group, key) for key in groups.keys()]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Building mosaics"):
            results.append(future.result())


    # Build one large mask mosaic across all mask COGs
    # >> All-region mask mosaic: union of every per-group mask COG, extent = their union.
    mask_cogs = [r[0] for r in results if r[1] == "mask"]

    if mask_cogs:
        # Extract unique date range (assuming all identical)
        start_dates = {r[5] for r in results if r[1] == "mask"}
        end_dates = {r[6] for r in results if r[1] == "mask"}

        # >> Refuses to mix periods in one mosaic.
        if len(start_dates) == 1 and len(end_dates) == 1:
            start_date = start_dates.pop()
            end_date = end_dates.pop()
        else:
            raise ValueError("Mask groups have inconsistent date ranges.")

        # >> Gotcha: big_vrt and tmp_tif are never used (leftovers).
        big_vrt = Path(output_dir) / "big_mask.vrt"
        tmp_tif = Path(output_dir) / "big_mask_tmp.tif"
        big_mask_path = (Path(output_dir)
                         / f"mining_mask_{start_date}_{end_date}_epsg4326.tif")
        # >> Writes mining_mask_<start>_<end>_epsg4326.tif covering all groups.
        build_cog(
            input_files=mask_cogs,
            output_path=big_mask_path,
            raster_type="mask"
        )
        # utm_zone/lat_start/lat_end are None: this mosaic spans all of them.
        results.append(
            (big_mask_path, "mask", None, None, None, start_date, end_date))

    # Build one large logits mosaic across all logits COGs
    # >> Same all-region mosaic for logits (only if logits were requested).
    logit_cogs = [r[0] for r in results if r[1] == "logits"]

    if logit_cogs:
        # Extract unique date range (assuming all identical)
        start_dates = {r[5] for r in results if r[1] == "logits"}
        end_dates = {r[6] for r in results if r[1] == "logits"}

        if len(start_dates) == 1 and len(end_dates) == 1:
            start_date = start_dates.pop()
            end_date = end_dates.pop()
        else:
            raise ValueError("Logits groups have inconsistent date ranges.")

        big_logits_path = (Path(output_dir)
                           / f"mining_logits_{start_date}_{end_date}_epsg4326.tif")
        build_cog(
            input_files=logit_cogs,
            output_path=big_logits_path,
            raster_type="logits"
        )
        # utm_zone/lat_start/lat_end are None: this mosaic spans all of them.
        results.append(
            (big_logits_path, "logits", None, None, None, start_date, end_date))

    # >> STAC = SpatioTemporal Asset Catalog, a JSON standard for listing geo files with
    # >> their footprint, dates and links. One "Feature" item per COG built above.
    stac_items = []

    for cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date in results:
        # >> Read the COG's bounds for the item geometry and bbox.
        with rasterio.open(cog_path) as ds:
            bounds = ds.bounds

        properties = {
            "utm_zone": utm_zone,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "raster_type": raster_type,
            "start_date": start_date,
            "end_date": end_date
        }

        # >> Gotcha: end_datetime is stamped end_date 23:59:59, but Earth Engine's end
        # >> date is exclusive, so that last day is not actually in the composite.
        if start_date and end_date:
            properties["start_datetime"] = f"{start_date}T00:00:00Z"
            properties["end_datetime"] = f"{end_date}T23:59:59Z"

        stac_items.append({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": Path(cog_path).name,
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
                    # str(): the catalog is JSON, and a Path is not serializable
                    "href": str(cog_path),
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                }
            }
        })

    stac_catalog = {
        "type": "FeatureCollection",
        "stac_version": "1.0.0",
        "description": "Amazon Mining Watch Scar Masks & Logits",
        "features": stac_items
    }

    # >> Write stac_catalog.json (a FeatureCollection of STAC items).
    with open(stac_out, "w") as f:
        json.dump(stac_catalog, f, indent=2)

    print("All processing complete.")

# >> CLI: python sam2_build_cog.py <run folder> [--raster_types mask logits] ...
# >> Defaults put outputs in <input_dir>/cog_outputs/.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build mine scar segmentation COGs from tile GeoTiffs.")
    parser.add_argument("input_dir", help="Directory containing EPSG:4326 DLTile GeoTIFFs")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/cog_outputs)")
    parser.add_argument("--index_out", default=None, help="Tile index GeoParquet path")
    parser.add_argument("--stac_out", default=None, help="STAC catalog output JSON")
    parser.add_argument(
        "--max_workers", type=int,
        default=min(DEFAULT_MAX_WORKERS, os.cpu_count() or DEFAULT_MAX_WORKERS),
        help=("Parallel worker count. Bounded by memory rather than CPU: each "
              "group holds a GDAL cache and a full-raster COG translate."))
    parser.add_argument(
        "--raster_types", nargs="+", choices=("mask", "logits"),
        default=["mask"],
        help=(
            "Which mosaics to build. Logit mosaics are for inspection only: "
            "re-deriving a mask at a different threshold must replay the "
            "smoothing per tile, because max-reduce across overlapping tiles "
            "does not commute with it. The durable logit artifact is the "
            "per-tile *-logits.tif, not the mosaic. Passing 'mask' alone skips "
            "the float32 half of the work, which dominates the run. Default is "
            "mask only; pass 'mask logits' if you want a logit mosaic to "
            "look at."
        ))
    parser.add_argument(
        "--extent_mode", choices=("band", "union"), default="union",
        help=(
            "Output extent for per-band mosaics. 'union' (default) uses the "
            "snapped union of the tiles present -- on the global lattice, so "
            "temporal rules stay valid with an integral pixel offset between "
            "periods. 'band' uses the fixed UTM-zone x lat-band box, giving "
            "byte-identical grids across periods, but writes 6.2 Gpx per band "
            "regardless of content: only ~1.1x the union on the dense core "
            "bands, but 16x on utm20 lat[-24,-16] and 79,000x on utm19 "
            "lat[-24,-16], whose real extent is 285x275 pixels. That has "
            "exhausted memory in gdalwarp on a loaded machine. Prefer 'band' "
            "only when identical extents are worth the cost and the run has "
            "the machine to itself."
        ))

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = Path(args.output_dir or Path(input_dir) / "cog_outputs")
    index_out = args.index_out or output_dir / "tile_index.parquet"
    stac_out = args.stac_out or output_dir / "stac_catalog.json"

    main(input_dir, output_dir, index_out, stac_out, args.max_workers,
         extent_mode=args.extent_mode, raster_types=tuple(args.raster_types))
