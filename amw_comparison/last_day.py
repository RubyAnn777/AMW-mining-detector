"""Do our score gaps with AMW lie under the scenes of the quarter's last day?

    python amw_comparison/last_day.py --run tests_output/aca_full_2026-10-23 --quarter 2025Q1

Our quarter ends on the 1st of the next month, excluded (Period.end). AMW's ends a day
earlier: its driver passes the quarter's last day as the end date (pipeline.py:80-81 via
core/periods.py), and Earth Engine excludes the end. So scenes taken on the last day
(31 Mar, 30 Jun, 30 Sep, 31 Dec) are in our composite only. If that is the whole
difference, chips scored differently lie under such a scene, and chips elsewhere are
identical.

Reads compare.py's results (<run>/amw_comparison/chips_<quarter>.parquet). Small Earth
Engine requests: the last day's scene list and outlines, no imagery. Writes nothing.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import ee
import pandas as pd
import shapely
from shapely.geometry import shape

import compare
import gee  # AMW's Earth Engine login (core/gee.py; compare.py put core/ on the path)
from compare import REPO
from periods import Period  # ours: panel_pipeline/lib/periods.py

S2 = "COPERNICUS/S2_HARMONIZED"   # gee.py:201


def windows(quarter: str) -> tuple[date, date, date]:
    """(start, AMW's end, our end); Earth Engine excludes both ends."""
    period = Period.parse(quarter)
    return period.start, period.end - timedelta(days=1), period.end


def login(run: Path) -> None:
    """Earth Engine, billed to the run's own project (never AMW's default)."""
    gee.EE_PROJECT = compare.frozen_config(run)["earth_engine"]["project"]
    gee._ensure_earth_engine_initialized()           # AMW's login (gee.py:47)


def last_day_scenes(chips: pd.DataFrame, quarter: str):
    """The Sentinel-2 scenes taken on the quarter's last day over the chips' box: a table
    (scene, cloud %) and their outline (shapely)."""
    box = ee.Geometry.Rectangle([chips["amw_key_x"].min(), chips["amw_key_y"].min(),
                                 chips["amw_key_x"].max(), chips["amw_key_y"].max()])
    _, amw_end, our_end = windows(quarter)
    col = ee.ImageCollection(S2).filterDate(amw_end.isoformat(), our_end.isoformat()).filterBounds(box)
    feats = col.map(lambda img: ee.Feature(img.geometry(), {
        "scene": img.get("system:index"), "cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE")})
    ).getInfo()["features"]
    scenes = pd.DataFrame([f["properties"] for f in feats], columns=["scene", "cloud_pct"])
    outline = shapely.union_all([shape(f["geometry"]) for f in feats]) if feats else shapely.Polygon()
    return scenes, outline


def under_last_day(chips: pd.DataFrame, outline) -> pd.DataFrame:
    """Per outcome: how many chips lie under the last day's scenes, and how many elsewhere."""
    t = chips.assign(outcome=compare.outcome(chips),
                     under=shapely.contains_xy(outline, chips["amw_key_x"], chips["amw_key_y"]))
    differ = (t["outcome"] == "both") & ((t["confidence"] - t["amw_confidence"]).abs() > 1e-6)
    groups = {o: t[t["outcome"] == o] for o in compare.OUTCOMES[:3]} | {"both, scores differ": t[differ]}
    table = pd.DataFrame({"under the last day's scenes": [int(g["under"].sum()) for g in groups.values()],
                          "elsewhere": [int((~g["under"]).sum()) for g in groups.values()]},
                         index=list(groups))
    table["share under"] = table.iloc[:, 0] / table.sum(axis=1)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score gaps against the last day's scenes.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--quarter", required=True, help="e.g. 2025Q1")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    chips = pd.read_parquet(run / "amw_comparison" / f"chips_{args.quarter}.parquet")
    login(run)
    scenes, outline = last_day_scenes(chips, args.quarter)
    print(f"{args.quarter}: {len(scenes)} Sentinel-2 scenes taken on {windows(args.quarter)[1]} "
          f"(in our quarter only), mean cloud {scenes['cloud_pct'].mean():.0f}%")
    if scenes.empty:
        raise SystemExit("no scene that day: the last day cannot explain any gap")
    print(under_last_day(chips, outline).to_string(float_format="{:.1%}".format))
