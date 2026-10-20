"""python amw_comparison/independent_check.py

Independent re-check of amw_comparison: matches AMW rows to our chips by
geometry (overlap of the squares), not by the rounded-centre key, and counts
everything inside our run's footprint, not only rows that hit a planned chip."""
import sys
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd, shapely

REPO = Path(__file__).resolve().parent.parent
OUT, AMW = REPO / "tests_output", REPO / "amw_comparison/data/amw"
RUNS = {"test_region_2026-10-19": 2, "test_region_para_2026-10-19": 4}
TAG = {"2026Q1": "Q126", "2026Q2": "Q226", "2026Q3": "Q326"}

amw_all = {q: gpd.read_file(AMW / f"amazon_basin_{t}_t0.4.geojson") for q, t in TAG.items()}
for run, sub in RUNS.items():
    base = OUT / run
    chips = pd.read_parquet(base / "plan" / "chips.parquet")
    cg = gpd.GeoDataFrame(chips, geometry=shapely.box(chips.minx, chips.miny, chips.maxx, chips.maxy), crs=4326)
    footprint = cg.union_all()
    print(f"\n##### {run} (subregion {sub}): {len(chips)} chips, in_region False: {(~chips.in_region).sum()}")
    for q in TAG:
        ours = pd.read_parquet(base / "detect/quarterly" / f"period={q}" / f"subregion={sub}" / "part_00001.parquet")
        t = cg.merge(ours, on="chip_id", validate="one_to_one")
        amw = amw_all[q]
        # every AMW row whose square lies mostly inside our footprint
        inside = amw[amw.intersects(footprint)].copy()
        frac = inside.intersection(footprint).area / inside.area
        inside = inside[frac > 0.5].reset_index(drop=True)
        # geometric match: best-overlapping chip
        j = gpd.sjoin(inside, cg[["chip_id", "geometry"]], predicate="intersects").reset_index()
        j["iou"] = [a.intersection(b).area / a.union(b).area for a, b in
                    zip(inside.geometry.loc[j["index"]], cg.set_index("chip_id").geometry.loc[j.chip_id])]
        best = j.sort_values("iou").groupby("index").tail(1).set_index("index")
        inside["chip_id"] = best["chip_id"]; inside["iou"] = best["iou"]
        good = inside[inside.iou > 0.9]
        dup = good.chip_id.duplicated().sum()
        m = t[["chip_id", "confidence"]].merge(good[["chip_id", "confidence"]].rename(columns={"confidence": "amw"}),
                                              on="chip_id", how="left")
        o = m.confidence > 0.4; a = m.amw.notna()
        both = m[o & a]; gap = (both.confidence - both.amw).abs()
        print(f"{q}: AMW rows mostly inside footprint {len(inside)} | IoU>0.9 {len(good)} (min IoU {inside.iou.min():.3f}, dup {dup}) "
              f"| ours>0.4 {o.sum()} | both {(o&a).sum()} onlyOurs {(o&~a).sum()} onlyAMW {(~o&a).sum()} "
              f"| AMW conf<=0.4 {(good.confidence<=0.4).sum()} | gap: max {gap.max():.2e} median {gap.median():.2e} "
              f"exact(<=1e-6) {(gap<=1e-6).mean():.0%}")
        dis = m[(o & ~a) | (~o & a)]
        if len(dis): print("   disagreements:\n", dis.to_string())
