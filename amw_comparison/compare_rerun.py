"""Compare several sets of raw detections for one quarter, chip by chip, on our run's chips.

    python amw_comparison/compare_rerun.py --run tests_output/test_region_para_2026-10-19 \
        --subregion 4 --quarter 2026Q1 \
        published=amw_comparison/data/amw/amazon_basin_Q126_t0.4.geojson \
        amw_code_0331=data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/<file ending _2026-03-31.geojson> \
        amw_code_0401=data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/<file ending _2026-04-01.geojson>

Each NAME=PATH is an AMW-style GeoJSON (confidence > 0.4 per chip). Our own detect.py
scores for the quarter are added as 'ours'. For every pair it prints how many chips each
side detects, how many both detect, and how many shared chips have identical scores.
Reads only.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "core"))
from postprocess import centroid_key  # noqa: E402  (AMW's chip key)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--subregion", type=int, required=True)
ap.add_argument("--quarter", required=True)
ap.add_argument("files", nargs="+", help="NAME=PATH")
args = ap.parse_args()
run = REPO / args.run

chips = pd.read_parquet(run / "plan/chips.parquet", columns=["chip_id", "amw_key_x", "amw_key_y"])
folder = run / "detect/quarterly" / f"period={args.quarter}" / f"subregion={args.subregion}"
ours = pd.concat(pd.read_parquet(p, columns=["chip_id", "confidence"]) for p in sorted(folder.glob("part_*.parquet")))
scores = {"ours": chips.merge(ours, on="chip_id").set_index("chip_id")["confidence"]}
scores["ours"] = scores["ours"][scores["ours"] > 0.4]

for item in args.files:
    name, path = item.split("=", 1)
    gdf = gpd.read_file(REPO / path if not Path(path).is_absolute() else path)
    gdf["amw_key_x"], gdf["amw_key_y"] = centroid_key(gdf)
    m = chips.merge(gdf[["amw_key_x", "amw_key_y", "confidence"]], on=["amw_key_x", "amw_key_y"])
    scores[name] = m.set_index("chip_id")["confidence"]

rows = []
for a, b in combinations(scores, 2):
    sa, sb = scores[a], scores[b]
    both = sa.index.intersection(sb.index)
    gap = (sa[both] - sb[both]).abs()
    rows.append({"pair": f"{a} vs {b}", "A": len(sa), "B": len(sb), "both": len(both),
                 "only A": len(sa.index.difference(sb.index)), "only B": len(sb.index.difference(sa.index)),
                 "identical (<=1e-6)": f"{(gap <= 1e-6).mean():.0%}" if len(both) else "-",
                 "largest gap": round(gap.max(), 4) if len(both) else None})
print()
print(pd.DataFrame(rows).to_string(index=False))
