"""python amw_comparison/tile_breakdown.py test_region_para_2026-10-19 4

Per-tile score gaps (ours vs AMW) for chips both detected, by quarter, plus vote check."""
import re, sys
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "core"))
from postprocess import centroid_key
OUT, AMW = REPO / "tests_output", REPO / "amw_comparison/data/amw"
run, sub = sys.argv[1], int(sys.argv[2])
TAG = {"2026Q1": "Q126", "2026Q2": "Q226", "2026Q3": "Q326"}
chips = pd.read_parquet(OUT / run / "plan/chips.parquet")
for q, t in TAG.items():
    amw = gpd.read_file(AMW / f"amazon_basin_{t}_t0.4.geojson")
    amw["amw_key_x"], amw["amw_key_y"] = centroid_key(amw)
    ours = pd.read_parquet(OUT / run / "detect/quarterly" / f"period={q}" / f"subregion={sub}" / "part_00001.parquet")
    d = chips.merge(ours, on="chip_id").merge(amw[["amw_key_x", "amw_key_y", "confidence", "preds"]].rename(columns={"confidence": "amw"}),
                                              on=["amw_key_x", "amw_key_y"], how="left")
    d["tile"] = d.tile_key
    b = d[d.amw.notna()].copy()
    b["gap"] = (b.confidence - b.amw).abs()
    v = np.array([[float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p)] for p in b.preds])
    vg = np.abs(b[[f"vote_{i}" for i in range(1, 7)]].to_numpy() - v).max()
    per = d.groupby("tile").agg(ours_det=("confidence", lambda c: (c > .4).sum()), amw_det=("amw", "count"),
                                n_obs=("n_obs", "median"), n_clear=("n_clear", "median"))
    per["max_gap"] = b.groupby("tile").gap.max()
    print(f"\n== {run} {q}: max vote gap {vg:.2e}; tiles where detections exist:")
    print(per[(per.ours_det > 0) | (per.amw_det > 0)].to_string())
    print("all tiles median n_obs", d.n_obs.median(), "n_clear", d.n_clear.median())
