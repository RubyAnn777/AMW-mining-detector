
# =============================================================================
# FILE OVERVIEW — scripts/concatenate.py
#
# Merge GeoJSON FeatureCollections into a single file. Rewritten upstream
# (July 2026) with geopandas: each input is read into a GeoDataFrame and the
# rows are pooled with pd.concat. No geometry processing, no deduplication,
# no reprojection — purely row concatenation. Inputs must share a CRS
# (geopandas errors on a mismatch).
#
# This replaces the old `geojson`-package implementation, which required a
# legacy top-level 'crs' member in every input and raised KeyError on
# RFC 7946-compliant files that lack one. That limitation is gone.
#
# Pipeline role: utility for combining per-region inference outputs
# (chip-square mine detections with a "confidence" property) into one file,
# e.g. before dissolving/thresholding with postprocess.py or dissolve.py.
#
# Invocation:
#   python scripts/concatenate.py a.geojson b.geojson c.geojson \
#       --outpath combined_detections.geojson
#
# Inputs on disk : one or more GeoJSON FeatureCollections.
# Outputs on disk: one merged GeoJSON at --outpath
#   (default ./combined_detections.geojson; parent dirs created if missing).
# =============================================================================

import argparse           # CLI parsing
from pathlib import Path  # output-directory creation

import geopandas as gpd   # geo-aware I/O; does all reading/writing here


def concatenate(paths, outpath):
    """Concatenate FeatureCollections.

    Reads each input into a GeoDataFrame and stacks the rows into one frame.

    Inputs:
        paths (list[str]): paths of GeoJSON FeatureCollections to merge;
            features are typically chip-square detections in EPSG:4326.
        outpath (str): path for the merged output GeoJSON.
    Returns: None.
    Side effects: reads each input; creates outpath's parent directory if
        needed; writes the merged collection to ``outpath``.
    Raises:
        ValueError: if ``paths`` is empty (or inputs disagree on CRS).
    Role in pipeline: pre-step to postprocessing when inference was run in
        several regional batches.
    """
    if not paths:
        raise ValueError("At least one input path is required.")

    # Read every input and stack rows; ignore_index renumbers 0..N-1.
    gdf = gpd.pd.concat([gpd.read_file(p) for p in paths], ignore_index=True)
    # Ensure the output directory exists before writing.
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(outpath, driver="GeoJSON", index=False)


if __name__ == '__main__':

    # CLI: positional input paths plus an optional output path.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*",
        help="Names of files with model detections to concatenate.",
        default=[])  # zero or more input GeoJSONs (positional args)
    parser.add_argument(
        "--outpath", type=str,
        help="Path to write concatenated detections",
        default='./combined_detections.geojson')  # default output filename

    args = parser.parse_args()   # parse CLI into a Namespace
    concatenate(**vars(args))    # forward args as keyword arguments
