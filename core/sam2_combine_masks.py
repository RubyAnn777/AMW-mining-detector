#!/usr/bin/env python3
"""
Combine two segmentation mask COGs with OR logic.

- Output value 1 where either input has 1 (mask).
- Output value 0 where the logical OR is 0 (both 0, or 0 vs nodata).
- Output nodata (2) only where both inputs are nodata (2).
- Areas covered by only one raster pass through that raster's value.

Uses a VRT-derived band so processing is streaming (no full-raster load).
base_mask.tif defines output resolution and grid; update.tif is warped to match.

"""

import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import rasterio

# >> FILE OVERVIEW: small standalone tool that ORs two mask COGs into one (upstream AMW
# >> file). Typical use: add an "update" mask (e.g. a new quarter's increment) onto a
# >> "base" mask. Not called by scripts/pipeline.py; run by hand. Same VRT pixel-function
# >> trick as sam2_build_cog.py. VRT = GDAL virtual raster (XML recipe, no pixel copy);
# >> COG = Cloud-Optimized GeoTIFF. Mask values: 1 = mine, 0 = not mine, 2 = nodata. Env:
# >> 4 GDAL threads; GDAL_VRT_ENABLE_PYTHON=YES lets GDAL run the embedded Python code.
# Cap threads at 4: the VRT read through the Python pixel function is effectively
# single-threaded, so extra warp threads add little benefit.
# GDAL_VRT_ENABLE_PYTHON is required when gdalwarp evaluates the derived VRT.
os.environ.setdefault("GDAL_NUM_THREADS", "4")
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")


# >> Prints a shell command and runs it; raises CalledProcessError if it fails.
# >> Gotcha (real bug): " ".join(cmd) needs strings, but main passes Path objects
# >> (stack_vrt, derived_vrt), so the first call raises TypeError before GDAL runs.
def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


# >> Main job: base mask + update mask -> one COG where 1 wins, 2 only if both are 2.
# >> Inputs: two mask GeoTIFF/COG paths (uint8, EPSG:4326). Output: COG at output_path.
# >> Grid: pixel size copied from the base mask, extent = union of both.
def main(base_mask_path: str, update_path: str, output_path: str) -> None:
    # >> Nodata value for masks ("not observed"), as in sam2_logits.MASK_NODATA.
    nodata_val = 2

    # --- Resolution and grid from base_mask.tif (rasterio; matches GDAL geotransform) ---
    # >> Read only the pixel size (degrees) from the base file's affine transform.
    with rasterio.open(base_mask_path) as ds:
        xres = abs(ds.transform.a)
        yres = abs(ds.transform.e)
    print(f"Output resolution (from base_mask): {xres}, {yres}")

    # >> Temporary folder for the two VRT XML files; deleted at the end.
    with tempfile.TemporaryDirectory(prefix="combine_masks_") as tmpdir:
        stack_vrt = Path(tmpdir) / "stack.vrt"
        derived_vrt = Path(tmpdir) / "stack_calc.vrt"

        # --- Build aligned stack: both rasters warped to same grid ---
        # >> gdalbuildvrt -separate: base -> band 1, update -> band 2, on a common grid.
        # >> -resolution user -tr: use the base pixel size; -tap: "target aligned pixels",
        # >> snap the extent to multiples of that size so pixel edges fall on the lattice.
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
        # >> Swap the two plain bands for one derived band computed from both.
        tree = ET.parse(stack_vrt)
        root = tree.getroot()

        for band in root.findall("VRTRasterBand"):
            root.remove(band)

        derived = ET.SubElement(root, "VRTRasterBand", {
            "dataType": "Byte",
            "band": "1",
            "subClass": "VRTDerivedRasterBand",
        })
        ET.SubElement(derived, "NoDataValue").text = str(nodata_val)
        ET.SubElement(derived, "PixelFunctionType").text = "mask_or"
        ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
        # >> The Python code string below is embedded in the VRT and run by GDAL per
        # >> block: in_ar[0] = base (A), in_ar[1] = update (B); both 2 -> 2, either 1 ->
        # >> 1, else 0.
        code = ET.SubElement(derived, "PixelFunctionCode")
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
        stack_basename = Path(stack_vrt).name
        for band_index in (1, 2):
            src = ET.SubElement(derived, "SimpleSource")
            ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
            ET.SubElement(src, "SourceBand").text = str(band_index)

        # >> Write stack_calc.vrt (the derived VRT).
        tree.write(derived_vrt)

        # --- Warp derived VRT to final COG ---
        # >> gdalwarp renders the derived VRT into the final file: -of COG output format;
        # >> -r near nearest neighbour (never blend 0/1); -multi parallel warping;
        # >> -dstnodata 2; ZSTD compression, 512 px blocks, PREDICTOR=2 (integer
        # >> differencing before compression), BIGTIFF=YES (allow files > 4 GB).
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

    print("Done.", output_path)


# >> CLI: python sam2_combine_masks.py base.tif update.tif out.tif
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine two segmentation mask COGs with OR logic (base_mask.tif + update.tif -> output)."
    )
    parser.add_argument(
        "base_mask_path",
        type=str,
        help="Path to base mask COG (base_mask.tif)",
    )
    parser.add_argument(
        "update_path",
        type=str,
        help="Path to update mask COG (update.tif)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output combined COG",
    )
    args = parser.parse_args()
    # >> Passes the three positional args in order (base, update, output).
    main(*vars(args).values())
