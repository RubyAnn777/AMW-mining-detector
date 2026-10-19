"""Which Sentinel-2 scenes went into a quarter's composite: ours, and AMW's.

    python amw_comparison/scene_dates.py --run tests_output/test_region_2026-10-19 --quarter 2026Q3

Our quarter ends on the 1st of the next month, excluded (Period.end). AMW's ends a
day earlier: its driver passes the quarter's last day as the end date
(pipeline.py:80-81 via core/periods.py), and Earth Engine excludes the end. So the
scenes taken on the last day are in our composite only. Where one of them is
clear, our median differs from AMW's.

Lists every scene over the run's area in the quarter: when it was taken, when
Earth Engine ingested it (a scene ingested after AMW scored the quarter is also
ours only), its cloud share, and whether it has a Cloud Score+ image.
Small Earth Engine requests (metadata only, no imagery).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import ee
import pandas as pd

from compare import REPO, frozen_config, read_chips

sys.path.insert(0, str(REPO / "panel_pipeline"))
from periods import Period  # noqa: E402  (ours: panel_pipeline/periods.py)
import gee  # noqa: E402  (AMW's Earth Engine login; compare.py put core/ on the path)

S2 = "COPERNICUS/S2_HARMONIZED"                              # gee.py:201
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"     # gee.py:202


def windows(quarter: str) -> tuple[date, date, date]:
    """(start, AMW's end, our end); both ends are excluded by Earth Engine."""
    period = Period.parse(quarter)
    return period.start, period.end - timedelta(days=1), period.end


def area(run: Path) -> ee.Geometry:
    """The box around the run's chips."""
    chips = read_chips(run)
    return ee.Geometry.Rectangle([chips["minx"].min(), chips["miny"].min(),
                                  chips["maxx"].max(), chips["maxy"].max()])


def utc(times, unit: str = "ms"):
    """Earth Engine's whole-number times (since 1970) as dates; missing stays missing."""
    return pd.to_datetime(pd.to_numeric(times, errors="coerce"), unit=unit, utc=True)


def day(ts) -> str:
    return "none" if pd.isna(ts) else f"{ts:%Y-%m-%d}"


def _rows(collection: ee.ImageCollection, fields: dict) -> list[dict]:
    feats = collection.map(lambda img: ee.Feature(
        None, {name: img.get(prop) for name, prop in fields.items()})).getInfo()["features"]
    return [f["properties"] for f in feats]


def scenes(region: ee.Geometry, start: date, end: date) -> pd.DataFrame:
    """One row per Sentinel-2 scene over the region, start <= taken < end."""
    rows, scores = [], []
    month = start
    while month < end:                           # month by month: listings stay small
        nxt = min(date(month.year + month.month // 12, month.month % 12 + 1, 1), end)
        s, e = month.isoformat(), nxt.isoformat()
        rows += _rows(ee.ImageCollection(S2).filterDate(s, e).filterBounds(region),
                      {"scene": "system:index", "taken": "system:time_start",
                       "version": "system:version", "cloud_pct": "CLOUDY_PIXEL_PERCENTAGE"})
        scores += _rows(ee.ImageCollection(CLOUD_SCORE).filterDate(s, e).filterBounds(region),
                        {"scene": "system:index", "version": "system:version"})
        month = nxt
    table = pd.DataFrame(rows, columns=["scene", "taken", "version", "cloud_pct"])
    scored = pd.DataFrame(scores, columns=["scene", "version"])
    table["taken"] = utc(table["taken"])
    # system:version is when Earth Engine last wrote the asset, in microseconds
    table["ingested"] = utc(table["version"], "us")
    table = table.merge(scored.rename(columns={"version": "cs_version"}), on="scene", how="left")
    table["cloud_score_ingested"] = utc(table["cs_version"], "us")
    return (table.drop(columns=["version", "cs_version"])
                 .sort_values("taken", ignore_index=True))


def last_day_footprint(region: ee.Geometry, quarter: str):
    """The outline (shapely) of the scenes taken on the quarter's last day."""
    from shapely.geometry import shape
    import shapely
    _, amw_end, our_end = windows(quarter)
    col = ee.ImageCollection(S2).filterDate(amw_end.isoformat(), our_end.isoformat())
    feats = col.filterBounds(region).map(lambda img: ee.Feature(img.geometry())).getInfo()
    return shapely.union_all([shape(f["geometry"]) for f in feats["features"]])


def login(run: Path) -> None:
    gee.EE_PROJECT = frozen_config(run)["earth_engine"]["project"]   # never AMW's default
    gee._ensure_earth_engine_initialized()                           # AMW's login (gee.py:47)


def report(table: pd.DataFrame, quarter: str) -> str:
    start, amw_end, our_end = windows(quarter)
    last = table[table["taken"].dt.date == amw_end]
    lines = [f"{quarter}, Sentinel-2 scenes over the run's area",
             f"  ours  {start} to {our_end} (excluded): {len(table)} scenes",
             f"  AMW   {start} to {amw_end} (excluded): {len(table) - len(last)} scenes",
             f"  last day {amw_end}: {len(last)} scene(s), in our composite only"]
    for _, s in last.iterrows():
        lines.append(f"      {s['scene']}  cloud {s['cloud_pct']:.0f}%  "
                     f"Cloud Score+ {'yes' if pd.notna(s['cloud_score_ingested']) else 'NO'}")
    missing = table["cloud_score_ingested"].isna().sum()
    lines.append(f"  scenes without a Cloud Score+ image: {missing}")
    late = table[table["taken"].dt.date >= amw_end - timedelta(days=14)]
    lines.append("  the quarter's last two weeks, by acquisition date "
                 "(when Earth Engine ingested the scenes and their cloud scores):")
    for taken, g in late.groupby(late["taken"].dt.date):
        lines.append(f"      {taken}  {len(g):3d} scenes  cloud {g['cloud_pct'].mean():5.1f}%  "
                     f"ingested {day(g['ingested'].min())}..{day(g['ingested'].max())}  "
                     f"cloud score {day(g['cloud_score_ingested'].min())}.."
                     f"{day(g['cloud_score_ingested'].max())}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Which Sentinel-2 scenes went into a quarter.")
    ap.add_argument("--run", required=True, help="a planned run folder")
    ap.add_argument("--quarter", required=True, help="e.g. 2026Q3")
    args = ap.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else REPO / run
    login(run)
    start, _, our_end = windows(args.quarter)
    print(report(scenes(area(run), start, our_end), args.quarter))
