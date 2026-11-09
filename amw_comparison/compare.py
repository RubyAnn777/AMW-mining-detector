"""Our detect.py scores against AMW's published quarterly detections, chip by chip.

    python amw_comparison/compare.py --run tests_output/aca_full_2026-10-23 --quarters 2025Q1 2025Q2
    python amw_comparison/compare.py --run <run> --year 2026      every quarter of 2026 in the run

Works for any run: a small test area, one subregion, or the whole basin. Reads the run and
AMW's files (amw_comparison/data/amw/), and writes a small results folder,
<run>/amw_comparison/, that notebooks/amw_comparison.py shows:

    inputs.csv               one row per task read (frequency, period, subregion): complete?
    amw_files.csv            AMW's files compared, with their fingerprint (AMW replaces files)
    joins.csv                per quarter and subregion: AMW rows on our chips, rows that
                             should have matched, our chips that share a location key
    chips_<quarter>.parquet  every chip that our quarter, AMW, or one of our three months
                             detected, all scores side by side; border tiles once

The join: AMW's files carry no chip ID, only each chip's square. AMW recognises a chip by
its rounded centre (centroid_key, core/postprocess.py:113), and plan.py stored that same
key for every chip (amw_key_x/y in plan/chips.parquet). An AMW row is one of our chips
exactly when the two keys are equal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.append(str(REPO / "core"))
from postprocess import centroid_key  # noqa: E402  (AMW code)
sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))   # before core/: both have a periods.py
from periods import Period  # noqa: E402  (ours: panel_pipeline/lib/periods.py)

AMW_DIR = HERE / "data" / "amw"      # AMW's files, as you downloaded them (ignored by git)
RAW_THRESHOLD = 0.4                  # AMW keeps confidence > 0.4, strictly (inference_engine.py:803)
OUTCOMES = ["both", "only ours", "only AMW", "neither"]
KEY = ["amw_key_x", "amw_key_y"]


# ------------------------------------------------------------ reading

def amw_file(quarter: str, folder: Path = AMW_DIR) -> Path:
    """AMW's raw detections for a quarter: '2026Q3' -> amazon_basin_Q326_t0.4.geojson."""
    year, q = quarter.split("Q")
    return folder / f"amazon_basin_Q{q}{year[2:]}_t0.4.geojson"


def read_amw(path: Path) -> pd.DataFrame:
    """AMW's published raw detections: one row per chip, with its location key."""
    gdf = gpd.read_file(path, columns=["confidence"])
    key_x, key_y = centroid_key(gdf)               # AMW's key, as AMW computes it
    return pd.DataFrame({"amw_key_x": key_x, "amw_key_y": key_y,
                         "amw_confidence": gdf["confidence"].to_numpy(dtype=np.float64)})


def task_folder(run: Path, period: str, subregion: int, frequency: str = "quarterly") -> Path:
    """One detect.py task's folder; period is '2025Q1' or, for monthly, '2025-01'."""
    return run / "detect" / frequency / f"period={period}" / f"subregion={subregion}"


def read_ours(run: Path, period: str, subregion: int, frequency: str = "quarterly",
              cols=("chip_id", "confidence", "n_obs", "n_clear", "empty_share")) -> pd.DataFrame:
    """Our scores for one task, every chip: all part files (part_00000.parquet once merged)."""
    folder = task_folder(run, period, subregion, frequency)
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part files in {folder}; has detect.py run?")
    return pd.concat([pd.read_parquet(p, columns=list(cols)) for p in parts], ignore_index=True)


def read_status(run: Path, period: str, subregion: int, frequency: str = "quarterly") -> dict:
    return json.loads((task_folder(run, period, subregion, frequency) / "_status.json").read_text())


def read_chips(run: Path, subregion: int) -> pd.DataFrame:
    """Our planned chips of one subregion's tiles, filtered while reading
    (a whole-basin plan, about 146 M chips, never sits in memory at once)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    tiles = pd.read_csv(run / "plan" / "tiles.csv")
    numbers = pa.array([key_number(k) for k in tiles.loc[tiles["subregion"] == subregion, "tile_key"]],
                       type=pa.int64())
    keep = pc.is_in(pc.divide(pc.field("chip_id"), 1000), value_set=numbers)
    chips = ds.dataset(run / "plan" / "chips.parquet").to_table(
        columns=["chip_id", "minx", "maxx", *KEY], filter=keep).to_pandas()
    chips["tile"] = chips["chip_id"] // 1000      # the tile's number (grid.py, chip_ids)
    return chips


def key_number(tile_key: str) -> int:
    """A tile's number (chip_id // 1000) from its key, '552:12:10.0:21:60:-100' = size:pad:res:zone:column:row."""
    zone, column, row = (int(v) for v in tile_key.split(":")[3:])
    return (zone * 1000 + column + 500) * 10_000 + row + 5000


def frozen_config(run: Path) -> dict:
    """The run's own settings, as plan.py copied them."""
    return yaml.safe_load((run / "run" / "config.yaml").read_text())


def quarters_of_year(run: Path, year: int) -> list[str]:
    """The run's quarters in one year, e.g. ['2026Q1', '2026Q2', '2026Q3'] if it ends in September."""
    periods = frozen_config(run)["periods"]
    first, last = str(periods["first_month"]), str(periods["last_month"])
    quarters = [f"{year}Q{i}" for i in range(1, 5)]
    # "YYYY-MM" tags sort as text in time order, so a plain comparison is enough
    return [q for q in quarters
            if all(first <= m.tag <= last for m in Period.parse(q).months())]


def quarters_with_amw(run: Path, year: int, folder: Path = AMW_DIR) -> list[str]:
    """The run's quarters of one year that have an AMW file (AMW publishes quarters from 2025)."""
    in_run = quarters_of_year(run, year)
    quarters = [q for q in in_run if amw_file(q, folder).exists()]
    for q in in_run:
        if q not in quarters:
            print(f"{q}: no AMW file {amw_file(q, folder).name} in {folder}, skipped", flush=True)
    if not quarters:
        raise SystemExit(f"COMPARE ERROR: no quarter of {year} in the run has an AMW file in {folder}")
    return quarters


def write_merged(table: pd.DataFrame, path: Path, key: list[str]) -> None:
    """Write a results table, keeping the rows of earlier calls that this call didn't redo.

    So comparing 2025 after 2026 adds 2025's rows and leaves 2026's in the file; comparing
    2026 again replaces only 2026's rows.
    """
    if path.exists() and not table.empty:
        old = pd.read_csv(path)
        if set(key) <= set(old.columns):
            redone = pd.MultiIndex.from_frame(table[key].astype(str))
            stale = pd.MultiIndex.from_frame(old[key].astype(str)).isin(redone)
            table = pd.concat([old[~stale], table], ignore_index=True)
    table.sort_values(key, kind="mergesort").to_csv(path, index=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ joining and counting

def match(chips: pd.DataFrame, amw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Our chips with amw_confidence (NaN where AMW has no row), and AMW's rows that are none of our chips."""
    if chips.duplicated(KEY).any():
        raise ValueError("two of our chips share an AMW key; the join would be ambiguous")
    if amw.duplicated(KEY).any():
        raise ValueError("AMW's file lists a location twice; was it deduplicated?")
    joined = chips.merge(amw, on=KEY, how="left", validate="one_to_one")
    ours = pd.MultiIndex.from_frame(chips[KEY])
    return joined, amw[~pd.MultiIndex.from_frame(amw[KEY]).isin(ours)]


def should_have_matched(unmatched: pd.DataFrame, chips: pd.DataFrame) -> pd.Series:
    """AMW rows centred among our chip centres that still matched none of them.

    Each tile's chip centres form a regular grid; an AMW centre inside it (to within half
    a step) must be one of its points, so anything counted here means the grids differ.
    Rows just outside, e.g. a neighbouring tile's chips over our edge, are not counted.
    """
    if unmatched.empty:
        return pd.Series(False, index=unmatched.index)
    half_step = (chips["maxx"] - chips["minx"]).median() / 4   # chip 48 px, step 24 px
    zones = chips.groupby("tile").agg(x0=("amw_key_x", "min"), x1=("amw_key_x", "max"),
                                      y0=("amw_key_y", "min"), y1=("amw_key_y", "max"))
    boxes = shapely.box(zones["x0"] - half_step, zones["y0"] - half_step,
                        zones["x1"] + half_step, zones["y1"] + half_step)
    points = shapely.points(unmatched["amw_key_x"], unmatched["amw_key_y"])
    hit, _ = shapely.STRtree(boxes).query(points, predicate="within")
    return pd.Series(np.isin(np.arange(len(unmatched)), hit), index=unmatched.index)


def outcome(table: pd.DataFrame, threshold: float = RAW_THRESHOLD) -> pd.Series:
    """Per chip: 'both', 'only ours', 'only AMW' or 'neither' detected it.
    AMW's file has only chips above its threshold, so no AMW row means at or below it."""
    ours = table["confidence"] > threshold
    amw = table["amw_confidence"] > threshold     # NaN > x is False
    labels = np.select([ours & amw, ours & ~amw, ~ours & amw], OUTCOMES[:3], OUTCOMES[3])
    return pd.Series(labels, index=table.index, name="outcome")


def agreement(table: pd.DataFrame, threshold: float = RAW_THRESHOLD) -> pd.Series:
    """The 2×2 counts, and the overlap of the two detection sets."""
    counts = outcome(table, threshold).value_counts().reindex(OUTCOMES, fill_value=0)
    either = counts["both"] + counts["only ours"] + counts["only AMW"]
    return pd.concat([counts, pd.Series({
        "ours total": counts["both"] + counts["only ours"],
        "AMW total": counts["both"] + counts["only AMW"],
        # chips both detected / chips either detected (1 = the same set)
        "overlap (Jaccard)": counts["both"] / either if either else np.nan,
    })])


def score_gap(table: pd.DataFrame) -> pd.Series:
    """How close the two scores are, over chips both sides detected."""
    both = table[outcome(table) == "both"]
    gap = (both["confidence"] - both["amw_confidence"]).abs()
    return pd.Series({
        "chips": len(both),
        "identical (gap <= 1e-6)": (gap <= 1e-6).mean(),
        "gap <= 0.01": (gap <= 1e-2).mean(),
        "median gap": gap.median(),
        "largest gap": gap.max(),
    })


def one_per_location(rows: pd.DataFrame) -> pd.DataFrame:
    """Each location once, the higher score, as AMW does (dedupe_detections, postprocess.py:146).

    Two of our chips share a location where tiles overlap (UTM zone seams, and border tiles,
    which both neighbouring subregions score). border_copies says how often a location was
    scored, border_spread how far those scores differ (0: the same images went in).
    """
    shared = rows.duplicated(KEY, keep=False)      # few rows: only these are grouped
    many = rows[shared]
    scores = many.groupby(KEY)["confidence"]
    many = (many.assign(border_copies=scores.transform("size"),
                        border_spread=scores.transform("max") - scores.transform("min"))
                .sort_values("confidence", ascending=False, kind="mergesort").drop_duplicates(KEY))
    return (pd.concat([rows[~shared].assign(border_copies=1, border_spread=0.0), many])
              .sort_values("chip_id", ignore_index=True))


# ------------------------------------------------------------ a whole run

def task_state(run: Path, frequency: str, period: str, subregion: int) -> dict:
    """What a task's _status.json says, or that it is missing."""
    row = {"frequency": frequency, "period": period, "subregion": subregion}
    try:
        s = read_status(run, period, subregion, frequency)
    except FileNotFoundError:
        return row | {"state": "missing"}
    return row | {"state": "complete" if s["complete"] else "incomplete",
                  "tiles": f"{s['tiles_saved']:,}/{s['tiles_planned']:,}",
                  "failed": len(s["tiles_failed"])}


def one_subregion(run, quarter, months, subregion, chips, amw, states):
    """The joined rows of one quarter and subregion that someone detected, and the join's counts."""
    ours = chips.merge(read_ours(run, quarter, subregion), on="chip_id", validate="one_to_one")
    within = one_per_location(ours.assign(subregion=subregion))   # UTM zone seams
    table, unmatched = match(within.drop(columns=["border_copies", "border_spread"]), amw)
    month_cols = []
    for month in months:
        if states[("monthly", month, subregion)]["state"] == "complete":
            scores = read_ours(run, month, subregion, "monthly", cols=("chip_id", "confidence"))
            table = table.merge(scores.rename(columns={"confidence": f"conf_{month}"}),
                                on="chip_id", how="left", validate="one_to_one")
            month_cols.append(f"conf_{month}")
    table["months_detected"] = (table[month_cols] > RAW_THRESHOLD).sum(axis=1).astype(int)
    table["months_scored"] = len(month_cols)
    join = {"quarter": quarter, "subregion": subregion,
            "AMW rows on our chips": int(table["amw_confidence"].notna().sum()),
            "should have matched": int(should_have_matched(unmatched, chips).sum()),
            "chips sharing a key": len(ours) - len(within)}
    detected = ((table["confidence"] > RAW_THRESHOLD) | (table["amw_confidence"] > RAW_THRESHOLD)
                | (table["months_detected"] > 0))
    return table.loc[detected].drop(columns=["minx", "maxx"]), join


def compare_run(run: Path, quarters: list[str], subregions: list[int] | None = None,
                amw_dir: Path = AMW_DIR) -> Path:
    """Join a run's quarters (and their months) with AMW's, every subregion; write the results folder."""
    out = run / "amw_comparison"
    out.mkdir(exist_ok=True)
    subregions = subregions or sorted(pd.read_csv(run / "plan" / "tiles.csv")["subregion"].unique())
    months = {q: [m.tag for m in Period.parse(q).months()] for q in quarters}

    states = {}
    for q in quarters:
        for n in subregions:
            states[("quarterly", q, n)] = task_state(run, "quarterly", q, n)
            for m in months[q]:
                states[("monthly", m, n)] = task_state(run, "monthly", m, n)
    write_merged(pd.DataFrame(states.values()), out / "inputs.csv",
                 ["frequency", "period", "subregion"])

    amw, files = {}, []
    for q in quarters:
        path = amw_file(q, amw_dir)
        if not path.exists():
            raise SystemExit(f"COMPARE ERROR: {path} is missing; download it from Source "
                             f"Cooperative (raw_detections/) into {path.parent}")
        amw[q] = read_amw(path)
        files.append({"quarter": q, "file": path.name, "rows": len(amw[q]), "sha256": sha256(path)[:12]})
    write_merged(pd.DataFrame(files), out / "amw_files.csv", ["quarter"])

    kept, joins = {q: [] for q in quarters}, []
    for n in subregions:                       # the plan is read once per subregion
        todo = [q for q in quarters if states[("quarterly", q, n)]["state"] == "complete"]
        if not todo:
            print(f"subregion {n}: no complete quarter, skipped", flush=True)
            continue
        t = time.time()
        chips = read_chips(run, n)
        for q in todo:
            rows, join = one_subregion(run, q, months[q], n, chips, amw[q], states)
            kept[q].append(rows)
            joins.append(join)
            print(f"subregion {n} {q}: {len(chips):,} chips, {len(rows):,} detected by someone, "
                  f"{join['should have matched']} should have matched "
                  f"({(time.time() - t) / 60:.1f} min)", flush=True)
        del chips

    write_merged(pd.DataFrame(joins, columns=["quarter", "subregion", "AMW rows on our chips",
                                              "should have matched", "chips sharing a key"]),
                 out / "joins.csv", ["quarter", "subregion"])
    for q, parts in kept.items():
        if parts:   # border tiles: in two subregions' lists, scored twice
            one_per_location(pd.concat(parts, ignore_index=True)).to_parquet(
                out / f"chips_{q}.parquet", index=False)
    print(f"results -> {out}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare a run with AMW's quarters, chip by chip.")
    ap.add_argument("--run", required=True, type=Path)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--quarters", nargs="+", help="e.g. 2025Q1 2025Q2")
    which.add_argument("--year", type=int, help="every quarter of this year in the run, e.g. 2026")
    ap.add_argument("--subregions", nargs="+", type=int, help="default: all in the plan")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    quarters = args.quarters or quarters_with_amw(run, args.year)
    print(f"quarters: {' '.join(quarters)}", flush=True)
    compare_run(run, quarters, args.subregions)
