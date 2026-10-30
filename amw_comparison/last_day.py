"""Do our score gaps with AMW lie under the scenes of the quarter's last day?

AMW's quarter leaves out its last day (31 Mar, 30 Jun, ...; see scene_dates.py).
If that day is the whole difference, chips scored differently lie under a scene
taken that day, and chips elsewhere have identical scores.

    python amw_comparison/last_day.py --results tests_output/aca_full_2026-10-23/amw_comparison \
        --quarter 2025Q1

Reads compare_run.py's chips_<quarter>.parquet. Small Earth Engine requests
(scene outlines only, no imagery). Writes nothing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import shapely
import yaml

import compare
import scene_dates
from compare import REPO


def under_last_day(chips: pd.DataFrame, footprint) -> pd.DataFrame:
    """Per outcome: chips under the last day's scenes, elsewhere, and how many scores are identical."""
    t = chips.assign(outcome=compare.outcome(chips))
    t["under"] = shapely.contains_xy(footprint, t["amw_key_x"], t["amw_key_y"])
    t["identical"] = (t["confidence"] - t["amw_confidence"]).abs() <= 1e-6
    t.loc[t["outcome"] != "both", "identical"] = False
    rows = [t[t["outcome"] == o] for o in compare.OUTCOMES[:3]]
    rows.append(t[(t["outcome"] == "both") & ~t["identical"]])
    table = pd.DataFrame({
        "under the last day's scenes": [int(r["under"].sum()) for r in rows],
        "elsewhere": [int((~r["under"]).sum()) for r in rows],
    }, index=compare.OUTCOMES[:3] + ["both, scores differ"])
    table["share under"] = table.iloc[:, 0] / table.sum(axis=1)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score gaps against the last day's scenes.")
    ap.add_argument("--results", required=True, type=Path, help="compare_run.py's output folder")
    ap.add_argument("--quarter", required=True, help="e.g. 2025Q1")
    ap.add_argument("--config", default="configs/aca_full_2026-10-23.yaml",
                    help="the run's config, for the Earth Engine project")
    args = ap.parse_args()

    results = args.results if args.results.is_absolute() else REPO / args.results
    chips = pd.read_parquet(results / f"chips_{args.quarter}.parquet")
    config = yaml.safe_load((REPO / args.config).read_text())
    scene_dates.gee.EE_PROJECT = config["earth_engine"]["project"]
    scene_dates.gee._ensure_earth_engine_initialized()

    region = scene_dates.ee.Geometry.Rectangle(
        [chips["amw_key_x"].min(), chips["amw_key_y"].min(),
         chips["amw_key_x"].max(), chips["amw_key_y"].max()])
    _, amw_end, our_end = scene_dates.windows(args.quarter)
    scenes = scene_dates.scenes(region, amw_end, our_end)
    print(f"{args.quarter}: {len(scenes)} Sentinel-2 scenes taken on {amw_end} over the compared "
          f"chips (in our quarter only), mean cloud {scenes['cloud_pct'].mean():.0f}%")
    if scenes.empty:
        raise SystemExit("no scene that day: the last day cannot explain any gap")
    footprint = scene_dates.last_day_footprint(region, args.quarter)
    print(under_last_day(chips, footprint).to_string(float_format="{:.1%}".format))
