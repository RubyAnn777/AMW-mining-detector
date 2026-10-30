"""Compare a big run (many subregions, monthly and quarterly) with AMW's quarters.

The heavy part, meant for the cluster: it reads every subregion's scores, joins
them with AMW's file chip by chip, and writes a small results folder that
compare_run.ipynb shows on the laptop.

    python amw_comparison/compare_run.py --run /work/goldmine/grambauer/runs/aca_full_2026-10-23 \
        --quarters 2025Q1 2025Q2

Writes <run>/amw_comparison/:
    inputs.csv          one row per task read (frequency, period, subregion): complete? failed tiles?
    joins.csv           per quarter and subregion: AMW rows on our chips, rows that should have matched
    chips_<quarter>.parquet
                        every chip that we (quarterly), AMW, or any of our three months detected,
                        with all scores side by side; border tiles once (the higher score, as AMW)

Needs AMW's file per quarter in amw_comparison/data/amw/ (amazon_basin_Q125_t0.4.geojson, ...).
Reads only from the run; writes only the results folder.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from compare import (RAW_THRESHOLD, REPO, amw_file, match, read_amw, read_chips, read_ours,
                     read_status, should_have_matched)

sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
from periods import Period  # noqa: E402  (ours: panel_pipeline/lib/periods.py)


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


def one_subregion(run, quarter, months, subregion, chips, amw, inputs):
    """The joined rows for one quarter and subregion, only chips someone detected."""
    ours = chips.merge(read_ours(run, quarter, subregion), on="chip_id", validate="one_to_one")
    # Two of our chips can share AMW's location key (the same ground in two tiles,
    # e.g. where UTM zones overlap). AMW keeps the higher score for a key
    # (dedupe_detections, postprocess.py:146), so we do the same before joining.
    ours = (ours.sort_values("confidence", ascending=False, kind="mergesort")
                .drop_duplicates(["amw_key_x", "amw_key_y"]))
    same_key = len(chips) - len(ours)
    table, unmatched = match(ours, amw)
    month_cols = []
    for month in months:
        if inputs[("monthly", month, subregion)]["state"] != "complete":
            continue
        scores = read_ours(run, month, subregion, frequency="monthly", cols=("chip_id", "confidence"))
        table = table.merge(scores.rename(columns={"confidence": f"conf_{month}"}),
                            on="chip_id", how="left", validate="one_to_one")
        month_cols.append(f"conf_{month}")
    table["months_detected"] = (table[month_cols] > RAW_THRESHOLD).sum(axis=1)
    table["months_scored"] = len(month_cols)
    off_grid = should_have_matched(unmatched, chips)
    join = {"quarter": quarter, "subregion": subregion,
            "AMW rows on our chips": int(table["amw_confidence"].notna().sum()),
            "should have matched": int(off_grid.sum()),
            "chips sharing a key, dropped": same_key}
    detected = ((table["confidence"] > RAW_THRESHOLD) | (table["amw_confidence"] > RAW_THRESHOLD)
                | (table["months_detected"] > 0))
    kept = table.loc[detected].drop(columns=["minx", "miny", "maxx", "maxy"])
    return kept.assign(subregion=subregion), join


def same_key_note(join) -> str:
    n = join["chips sharing a key, dropped"]
    return f"{n:,} chips sharing a location key dropped " if n else ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare a run with AMW's quarters, all subregions.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--quarters", nargs="+", required=True, help="e.g. 2025Q1 2025Q2")
    ap.add_argument("--subregions", nargs="+", type=int, help="default: all in the plan")
    args = ap.parse_args()

    run = args.run
    out = run / "amw_comparison"
    out.mkdir(exist_ok=True)
    subregions = args.subregions or sorted(pd.read_csv(run / "plan" / "tiles.csv")["subregion"].unique())
    months = {q: [m.tag for m in Period.parse(q).months()] for q in args.quarters}

    inputs = {}
    for q in args.quarters:
        for n in subregions:
            inputs[("quarterly", q, n)] = task_state(run, "quarterly", q, n)
            for m in months[q]:
                inputs[("monthly", m, n)] = task_state(run, "monthly", m, n)
    pd.DataFrame(inputs.values()).to_csv(out / "inputs.csv", index=False)

    amw = {}
    for q in args.quarters:
        path = amw_file(q)
        if not path.exists():
            raise SystemExit(f"COMPARE ERROR: {path} is missing; download it from Source "
                             f"Cooperative (raw_detections/) into {path.parent}")
        amw[q] = read_amw(path).drop(columns="amw_preds", errors="ignore")

    kept = {q: [] for q in args.quarters}
    joins = []
    for n in subregions:                       # the plan is read once per subregion
        todo = [q for q in args.quarters if inputs[("quarterly", q, n)]["state"] == "complete"]
        if not todo:
            print(f"subregion {n}: no complete quarter, skipped", flush=True)
            continue
        t = time.time()
        chips = read_chips(run, n)
        for q in todo:
            rows, join = one_subregion(run, q, months[q], n, chips, amw[q], inputs)
            kept[q].append(rows)
            joins.append(join)
            print(f"subregion {n} {q}: {len(chips):,} chips, {len(rows):,} detected by someone, "
                  f"{join['should have matched']} should have matched, "
                  f"{same_key_note(join)}"
                  f"({(time.time() - t) / 60:.1f} min)", flush=True)
        del chips

    pd.DataFrame(joins).to_csv(out / "joins.csv", index=False)
    for q, parts in kept.items():
        if not parts:
            continue
        rows = pd.concat(parts, ignore_index=True)
        # A border tile is in two subregions' lists and was scored twice; keep the
        # higher score per location, as AMW's concat does (postprocess.py:146).
        key = ["amw_key_x", "amw_key_y"]
        rows["border_copies"] = rows.groupby(key)["chip_id"].transform("size")
        rows = (rows.sort_values("confidence", ascending=False, kind="mergesort")
                    .drop_duplicates(key).sort_values("chip_id"))
        rows.to_parquet(out / f"chips_{q}.parquet", index=False)
        print(f"{q}: {len(rows):,} chips -> {out / f'chips_{q}.parquet'}", flush=True)
