#!/usr/bin/env python3
"""
Combine two segmentation mask COGs with OR logic.

- Output value 1 where either input has 1 (mask).
- Output value 0 where the logical OR is 0 (both 0, or 0 vs nodata).
- Output nodata (2) only where both inputs are nodata (2).
- Areas covered by only one raster pass through that raster's value.

Uses a VRT-derived band so processing is streaming (no full-raster load).
base_mask.tif defines output resolution and grid; update.tif is warped to match.

FILE OVERVIEW (expanded)
------------------------
Pipeline role: this is how the Amazon Mining Watch cumulative time-series
raster gets extended each quarter.  Mining scars accumulate -- once mined,
a pixel counts as "ever mined" -- so OR-merging the cumulative-through-Q2
mask with a Q3-only update yields the cumulative-through-Q3 mask.

Inputs on disk (both are basin-wide mask COGs from sam2_build_cog.py):
  * base_mask_path: cumulative mask COG (uint8; 1 = mine, 0 = not mine,
    nodata = 2), e.g. mining_mask_2023-01-01_2024-06-30_epsg4326.tif.
    Its pixel grid and resolution define the output.
  * update_path: newer-quarter mask COG with the same value convention;
    it is warped onto the base grid.

Output on disk:
  * output_path: merged mask COG (uint8, same 1/0/2 convention; ZSTD,
    512-px blocks, PREDICTOR=2) covering the union of both extents.

How the merge streams (never loads the full rasters):
  1. gdalbuildvrt -separate stacks the two inputs as bands 1 and 2 of a
     VRT on the base raster's grid (-tr from base, -tap alignment).
  2. This script rewrites that VRT's XML: the two plain bands are
     replaced by one VRTDerivedRasterBand whose pixel function is inline
     Python (requires env var GDAL_VRT_ENABLE_PYTHON=YES): output 1 if
     either input is 1; nodata 2 only where BOTH are nodata; else 0.
  3. gdalwarp pulls blocks through the derived VRT (running the pixel
     function per block) and writes the final COG.

Invocation example:
    python sam2_combine_masks.py \
        mining_mask_2023-01-01_2024-06-30_epsg4326.tif \
        mining_mask_2024-07-01_2024-09-30_epsg4326.tif \
        mining_mask_2023-01-01_2024-09-30_epsg4326.tif

Pipeline position: last stage, after sam2_mask.py (per-tile SAM2 masks)
and sam2_build_cog.py (COG mosaics) -- keeps the public cumulative
mining-mask series growing quarter over quarter.
"""

# argparse: minimal 3-positional-argument CLI (base, update, output).
import argparse
# os: environment variables (GDAL settings) and path manipulation.
import os
# subprocess: shells out to the gdalbuildvrt / gdalwarp CLI tools.
import subprocess
# tempfile: scratch dir holding the two intermediate VRT files.
import tempfile
# ElementTree: surgically rewrites the VRT's XML band definitions.
import xml.etree.ElementTree as ET
# pathlib imported for parity with sibling scripts (not used below).
from pathlib import Path

# Set before importing GDAL so VRT Python pixel functions and threading are enabled.
# Cap threads at 4: the VRT read through the Python pixel function is effectively
# single-threaded, so extra warp threads add little benefit.
os.environ.setdefault("GDAL_NUM_THREADS", "4")
# Security opt-in: without this, GDAL refuses to execute the inline
# Python pixel function embedded in the VRT and the merge would fail.
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")

# GDAL Python bindings; imported only after the env vars above are set.
from osgeo import gdal


def run(cmd):
    """Run an external command, echoing it first and raising on failure.

    Args:
        cmd (list[str]): argv-style command, e.g. ["gdalwarp", src, dst].

    Returns:
        None.

    Side effects:
        Prints the exact command line (audit trail), then executes it;
        subprocess.check_call raises CalledProcessError on non-zero exit.
        Unlike the run() in sam2_build_cog.py, output is NOT suppressed:
        gdalwarp progress is useful during these long basin-scale merges.
    """
    print("Running:", " ".join(cmd))  # show the exact GDAL invocation
    subprocess.check_call(cmd)  # blocks until done; raises on failure


def main(base_mask_path: str, update_path: str, output_path: str) -> None:
    """OR-merge two mask COGs into one cumulative mask COG.

    Purpose:
        Extend the cumulative "ever mined" raster: a pixel is 1 (mine)
        if either input marks it 1; 0 where the OR resolves to 0; and
        nodata (2) only where BOTH inputs are nodata.  The base raster
        defines the output grid; the update is resampled onto it.

    Args:
        base_mask_path (str): path to the base mask COG (uint8; 1=mine,
            0=not mine, nodata=2), e.g. the cumulative-through-Q2
            basin mosaic.  Its pixel size (degrees, EPSG:4326) and grid
            alignment define the output raster.
        update_path (str): path to the update mask COG (same value
            convention), e.g. the newest quarter's mosaic; warped onto
            the base grid with nearest-neighbour resampling.
        output_path (str): destination path for the merged COG.

    Returns:
        None.

    Side effects (files written):
        * output_path: merged uint8 COG (ZSTD, PREDICTOR=2, 512-px
          blocks, nodata=2) spanning the union of both input extents.
        * Temporary stack.vrt / stack_calc.vrt in a scratch directory,
          removed automatically when the context manager exits.

    Pipeline role:
        Quarterly final step after sam2_build_cog.py -- produces the
        next release of the cumulative mining-mask time series.
    """
    # Byte value marking "no valid observation" in every AMW mask raster.
    nodata_val = 2

    # --- Resolution and grid from base_mask.tif ---
    # Open the base COG just to read its geotransform (no pixel reads).
    ds = gdal.Open(base_mask_path)
    # GDAL geotransform tuple: (origin_x, px_width, row_rotation,
    # origin_y, col_rotation, px_height); px_height < 0 when north-up.
    gt = ds.GetGeoTransform()
    xres = abs(gt[1])  # pixel width in degrees (EPSG:4326)
    yres = abs(gt[5])  # pixel height in degrees (abs: stored negative)
    ds = None  # GDAL idiom: dropping the reference closes the dataset
    print(f"Output resolution (from base_mask): {xres}, {yres}")

    # Scratch space for the two intermediate VRTs; auto-deleted on exit.
    with tempfile.TemporaryDirectory(prefix="combine_masks_") as tmpdir:
        stack_vrt = os.path.join(tmpdir, "stack.vrt")         # 2-band stack
        derived_vrt = os.path.join(tmpdir, "stack_calc.vrt")  # OR-logic band

        # --- Build aligned stack: both rasters warped to same grid ---
        # -separate: keep inputs as separate bands (1=base, 2=update)
        #     rather than mosaicking them into a single band.
        # -resolution user + -tr: force the base raster's pixel size.
        # -tap: snap the extent to multiples of that resolution so both
        #     inputs land on identical, target-aligned pixel edges.
        run([
            "gdalbuildvrt",
            "-resolution", "user",
            "-tr", str(xres), str(yres),
            "-tap",
            "-separate",
            stack_vrt,
            base_mask_path,
            update_path,
        ])

        # --- Replace bands with a single derived band (OR + nodata logic) ---
        # A VRT is plain XML, so ElementTree can rewrite its structure.
        tree = ET.parse(stack_vrt)
        root = tree.getroot()

        # Drop the two ordinary bands gdalbuildvrt generated; the derived
        # band added below re-reads the same sources itself.
        for band in root.findall("VRTRasterBand"):
            root.remove(band)

        # Single output band computed on the fly by a pixel function
        # (subClass VRTDerivedRasterBand); Byte matches the mask dtype.
        derived = ET.SubElement(root, "VRTRasterBand", {
            "dataType": "Byte",
            "band": "1",
            "subClass": "VRTDerivedRasterBand",
        })
        ET.SubElement(derived, "NoDataValue").text = str(nodata_val)  # 2
        # Must match the function name defined in PixelFunctionCode below.
        ET.SubElement(derived, "PixelFunctionType").text = "mask_or"
        # Inline Python -- the reason GDAL_VRT_ENABLE_PYTHON=YES is set.
        ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
        code = ET.SubElement(derived, "PixelFunctionCode")
        # GDAL calls mask_or once per processing window: in_ar is a list
        # of numpy arrays (one per SimpleSource: A=base, B=update; uint8,
        # shape (ysize, xsize)); out_ar is filled in place.  Logic:
        # both nodata (2,2) -> 2; else 1 if either equals 1; else 0.
        # A pixel nodata in only ONE input therefore passes through the
        # other input's value, and 2s never leak into valid areas.
        code.text = """
import numpy as np
def mask_or(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    A = in_ar[0]
    B = in_ar[1]
    out_ar[:] = np.where(
        (A == 2) & (B == 2),
        2,
        np.where((A == 1) | (B == 1), 1, 0),
    )
"""
        # Reference the two bands of the stack VRT (filename relative to derived VRT)
        # Both temp VRTs sit in tmpdir, so a bare filename resolves fine.
        stack_basename = os.path.basename(stack_vrt)
        # One SimpleSource per input band: band 1 = base, band 2 = update;
        # this order fixes in_ar[0] / in_ar[1] inside mask_or above.
        for band_index in (1, 2):
            src = ET.SubElement(derived, "SimpleSource")
            ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
            ET.SubElement(src, "SourceBand").text = str(band_index)

        # Persist the modified XML as a second VRT beside the stack VRT.
        tree.write(derived_vrt)

        # --- Warp derived VRT to final COG ---
        # gdalwarp pulls blocks through the derived band (running mask_or
        # per block -- streaming, never the whole raster in memory) and
        # writes a COG directly: nearest resampling (categorical values),
        # -multi enables multithreaded warping, ZSTD + PREDICTOR=2
        # (integer horizontal differencing), 512-px blocks, nodata=2.
        run([
            "gdalwarp",
            derived_vrt,
            output_path,
            "-of", "COG",
            "-r", "near",
            "-multi",
            "-dstnodata", str(nodata_val),
            "-co", "COMPRESS=ZSTD",
            "-co", "BLOCKSIZE=512",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=YES",
        ])

    print("Done.", output_path)  # final path echo for logs / scripting


if __name__ == "__main__":
    # Three positional path arguments, in the order main() expects.
    parser = argparse.ArgumentParser(
        description="Combine two segmentation mask COGs with OR logic (base_mask.tif + update.tif -> output)."
    )
    # Cumulative mask so far; also defines the output grid/resolution.
    parser.add_argument(
        "base_mask_path",
        type=str,
        help="Path to base mask COG (base_mask.tif)",
    )
    # New quarter's mask; warped onto the base grid before merging.
    parser.add_argument(
        "update_path",
        type=str,
        help="Path to update mask COG (update.tif)",
    )
    # Where the merged cumulative COG is written.
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output combined COG",
    )
    args = parser.parse_args()
    # Namespace preserves argument insertion order, so the values align
    # with main()'s (base_mask_path, update_path, output_path) signature.
    main(*vars(args).values())
