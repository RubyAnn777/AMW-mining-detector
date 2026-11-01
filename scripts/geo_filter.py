
# >> FILE OVERVIEW
# >> Keeps only detections that touch a boundary polygon (e.g. the Andes supplemental
# >> area). Called by scripts/pipeline.py:stage_filter; its output is the
# >> raw_detections/andes_supplemental/ file that persistence.py unions in.
# >> Upstream Amazon Mining Watch file.
import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from postprocess import PostprocessConfig  # noqa: E402


# >> Clips detections to a boundary: keep rows whose geometry intersects it.
# >> In: detections GeoJSON, boundary GeoJSON. Out: GeoJSON at outpath (6 decimals).
# >> Gotcha: no CRS check; both files must share a CRS (EPSG:4326 in practice).
# >> Name 'filter' shadows Python's built-in filter() inside this module.
def filter(path, boundary_path, outpath):
    """Filter detections to those intersecting a boundary.

    The caller names the destination, so the filtered file lands where downstream
    stages read from. The input is left as the pristine inference output.
    """
    gdf = gpd.read_file(path)
    boundary = gpd.read_file(boundary_path)
    print(f'{len(gdf)} features prior to filtering')
    # >> union_all merges all boundary polygons into one shape; keep chips touching it.
    filtered = gdf[gdf.intersects(boundary.union_all())]
    print(f'{len(filtered)} features after filtering')
    # >> File write: filtered detections, GeoJSON, 6 decimals.
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    # Pinned so precision does not float with the GDAL/pyogrio version; see
    # core/postprocess.py and docs/design/persistence-planning.md.
    filtered.to_file(outpath, driver="GeoJSON", index=False,
                     COORDINATE_PRECISION=PostprocessConfig.coordinate_precision)
    print(f'Wrote {len(filtered)} detections to {outpath}')


if __name__ == '__main__':

    # >> CLI: detections path, boundary path, required --outpath.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str,
        help="Path to model detections")
    parser.add_argument(
        "boundary_path", type=str,
        help="Path to boundary GeoJSON")
    parser.add_argument(
        "--outpath", type=str, required=True,
        help=("Path to write filtered detections. Required: the destination is "
              "where downstream stages read from, so it is named, not guessed."))

    args = parser.parse_args()
    filter(**vars(args))
