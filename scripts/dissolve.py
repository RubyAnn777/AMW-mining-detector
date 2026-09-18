# PROVENANCE: FORK-MODIFIED — upstream file modified in this fork (comments/docstrings only); upstream original: earthrise-media/mining-detector

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from postprocess import PostprocessConfig  # noqa: E402

def dissolve(path, threshold, column, buffer_width):
    """Merge adjacent patch-wise mine detections into polygons.

    Optionally thresholds detections on a score column, then performs the
    buffered dissolve: inflate each patch by ``buffer_width`` so touching
    or overlapping neighbors merge, union everything, split the union into
    connected components, and deflate to undo the inflation.

    Inputs:
        path (str): input GeoJSON of patch detections; geometry in the
            file's CRS (typically EPSG:4326, i.e. DEGREES).
        threshold (float | None): if given, keep only rows whose
            ``column`` value is strictly greater than this (score in
            [0, 1]); None skips filtering.
        column (str): name of the score column to threshold (CLI default
            'prob'; current AMW inference files use 'confidence').
        buffer_width (float): buffer half-width in the CRS units of the
            file — DEGREES for EPSG:4326. Default 1e-5 deg (~1 m at the
            equator), just enough to bridge floating-point gaps between
            adjacent chips.

    Returns: None.
    Side effects: prints feature counts before/after thresholding; writes
        `<input stem>-dissolved-<threshold>.geojson` beside the input.
        NOTE: the output GeoDataFrame is built without an explicit CRS and
        keeps no attribute columns — confidences are NOT carried over
        (unlike postprocess.dissolve_patches, which averages them).
    Role in pipeline: standalone alternative to postprocess.py's dissolve
        stage.
    """
    df = gpd.read_file(path)  # load detections as a GeoDataFrame
    print(f'{len(df)} features prior to filtering')  # pre-filter count
    # Optional score gate: strict '>' comparison against the chosen column.
    if threshold is not None:
        df = df.loc[df[column] > threshold]
    print(f'{len(df)} features after to filtering')  # post-filter count

    # Buffered dissolve, steps 1-2: inflate each patch by buffer_width
    # (join_style=2 = mitred corners, keeps squares square), then
    # unary_union merges the whole layer into one (Multi)Polygon.
    # (unary_union is the legacy name for what union_all() does now.)
    dissolved = df.geometry.buffer(buffer_width, join_style=2).unary_union
    # Step 3: wrap in a one-row frame, explode the MultiPolygon into one row
    # per connected component; reset_index gives a clean 0..N-1 RangeIndex.
    # NOTE: no crs= is passed, so the frame carries no CRS metadata.
    df = gpd.GeoDataFrame(geometry=[dissolved]).explode().reset_index(drop=True)
    # Step 4: negative buffer undoes the inflation, restoring true outlines.
    df.geometry = df.geometry.buffer(-buffer_width, join_style=2)
    # Output name embeds the threshold when one was supplied. Upstream fix
    # (July 2026): plain '-dissolved' suffix otherwise, instead of
    # interpolating the literal string 'None'.
    base = path.split('.geojson')[0]
    suffix = f'-dissolved-{threshold}' if threshold is not None else '-dissolved'
    outpath = base + suffix + '.geojson'
    # Pinned so precision does not float with the GDAL/pyogrio version; see
    # core/postprocess.py and docs/design/persistence-planning.md.
    df.to_file(outpath, driver="GeoJSON", COORDINATE_PRECISION=PostprocessConfig.coordinate_precision)

if __name__ == '__main__':
    
    # CLI: one positional input path plus optional filtering/buffer knobs.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str,
        help="Path to model detections")  # input GeoJSON (positional)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Prediction threshold between [0, 1]; if given, the dataframe must have a 'pred' column'")
    # ^ help text is stale: the column actually used is --column (default
    #   'prob'), not 'pred'.
    parser.add_argument(
        "--column", type=str, default='prob',
        help="Name of prediction value to threshold")  # 'confidence' for AMW files
    parser.add_argument(
        "--buffer_width", type=float,
        help="Distance in dataframe CRS to buffer patches for smooth dissolve",
        default=0.00001)  # DEGREES for EPSG:4326 (~1 m at the equator)

    args = parser.parse_args()  # parse CLI into a Namespace
    dissolve(**vars(args))      # forward args as keyword arguments

    
