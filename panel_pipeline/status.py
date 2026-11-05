"""How far a run has got: one line per year, one mark per task.

    python panel_pipeline/status.py --run tests_output/test_region_small_2026-10-19
    python panel_pipeline/status.py --run ... --frequency quarterly
    python panel_pipeline/status.py --run ... --todo > tasks.txt   the commands still to run
    python panel_pipeline/status.py --run ... --all                 a progress bar for every task

Below the grid it shows a progress bar per task (counted in tiles, from _status.json,
which detect.py updates after every batch) for each task that has started but is not
done, and which Slurm job and node each running task is on (from its _lock.json).

A task is one detect/<frequency>/period=<tag>/subregion=<n>/ folder, written by one
detect.py command. This reads the run's frozen config, plan/tiles.csv, and each folder's
_status.json and _lock.json. It changes nothing, so it is safe while detect.py runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))   # our helpers: config, periods, grid, ...

import lock
from config import Config, load_config
from periods import Period

# ADDITION: AMW checks progress by reading tmux windows on its VMs; with hundreds of
# tasks on a cluster we need one overview.
MARKS = {"done": "#", "running": "R", "failed": "!", "partly done": "~",
         "stopped": "S", "not started": "."}
LEGEND = ("#  done   R  running   !  has failed tiles   ~  partly done   "
          "S  stopped without cleaning up   .  not started")
BAR = 20                      # characters in a progress bar
COLUMNS = {"monthly": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "quarterly": ["Q1", "Q2", "Q3", "Q4"]}


def tiles_per_subregion(run_dir: Path) -> Counter:
    """How many tiles each subregion has, the same in every period (plan/tiles.csv).

    Only the subregions this run scores are in it (a test's clip_to can leave some empty).
    """
    with open(run_dir / "plan" / "tiles.csv", newline="") as f:
        return Counter(int(row["subregion"]) for row in csv.DictReader(f))


def tiles_saved(folder: Path, planned: int) -> tuple[int, int]:
    """(tiles saved, tiles planned) of a task; 0 saved if it has no _status.json yet."""
    try:
        status = json.loads((folder / "_status.json").read_text())
        return status["tiles_saved"], status["tiles_planned"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0, planned


def job_of(folder: Path) -> str | None:
    """'job 1800424 on node12' for a running task (its process number outside Slurm)."""
    held = lock.read(folder)
    if held is None or held["stale"]:
        return None
    who = f"job {held['job']}" if held.get("job") else f"process {held.get('pid', '?')}"
    return f"{who} on {held.get('host', '?')}"


def bar(saved: int, planned: int) -> str:
    """[#########...........]  45%   (rounded down: a full bar means really done)"""
    share = saved / planned if planned else 0.0
    return f"[{'#' * int(share * BAR)}{'.' * (BAR - int(share * BAR))}] {int(share * 100):3d}%"


def task_state(folder: Path) -> tuple[str, str]:
    """A task's state, and what is worth knowing about it ("" if nothing)."""
    held = lock.read(folder)
    if held is not None and not held["stale"]:
        return "running", lock.who(held)
    if held is not None:
        return "stopped", (f"stopped without cleaning up ({lock.who(held)}); "
                           f"run its command again to resume")
    path = folder / "_status.json"
    if not path.is_file():
        return "not started", ""
    status = json.loads(path.read_text())
    if status["complete"]:
        return "done", ""
    if status["tiles_failed"]:
        key, error = next(iter(status["tiles_failed"].items()))
        return "failed", (f"{len(status['tiles_failed'])} tile(s) failed, "
                          f"e.g. {key}: {error[:120]}")
    limit = status["sessions"][-1].get("limit")
    return "partly done", (f"{status['tiles_saved']:,}/{status['tiles_planned']:,} tiles saved"
                           + (f" (last run used --limit {limit})" if limit else ""))


def tasks(cfg: Config, run_dir: Path, frequencies: list[str],
          planned: Counter) -> list[dict]:
    """Every task of the run, period first: 2018-01 subregions 1..6, then 2018-02, ..."""
    out = []
    for freq in frequencies:
        for period in cfg.periods[freq]:
            for n in sorted(planned):
                folder = (run_dir / "detect" / freq / f"period={period.tag}"
                          / f"subregion={n}")
                state, note = task_state(folder)
                saved, of = tiles_saved(folder, planned[n])
                out.append({"frequency": freq, "period": period, "subregion": n,
                            "state": state, "note": note, "saved": saved, "planned": of,
                            "job": job_of(folder) if state == "running" else None})
    return out


def grid(rows: list[dict], freq: str, subregions: list[int]) -> list[str]:
    """One line per year; each cell holds one mark per subregion, in subregion order."""
    marks: dict[tuple[int, int], str] = {}
    for row in rows:
        p: Period = row["period"]
        column = (p.first_month - 1) // (1 if freq == "monthly" else 3)
        marks[(p.year, column)] = marks.get((p.year, column), "") + MARKS[row["state"]]
    width = max(len(subregions), 3) + 1
    lines = [("      " + "".join(c.ljust(width) for c in COLUMNS[freq])).rstrip()]
    for year in sorted({y for y, _ in marks}):
        lines.append(f"{year}  " + "".join(marks.get((year, c), "").ljust(width)
                                           for c in range(len(COLUMNS[freq]))).rstrip())
    return lines


def command(run: str, row: dict) -> str:
    tag = row["period"].tag
    return (f"python panel_pipeline/detect.py --run {run} --frequency {row['frequency']} "
            f"--subregion {row['subregion']} --from {tag} --to {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show how far a run has got.")
    ap.add_argument("--run", required=True, help="the run folder made by plan.py")
    ap.add_argument("--frequency", choices=("monthly", "quarterly"),
                    help="only this frequency (default: all of the run's)")
    ap.add_argument("--todo", action="store_true",
                    help="print only the detect.py commands for tasks not done and not "
                         "running, period first, one per line")
    ap.add_argument("--all", action="store_true",
                    help="a progress bar for every task, also those done or not started")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()      # as detect.py reads --run
    if not (run_dir / "run" / "config.yaml").is_file():
        raise SystemExit(f"STATUS ERROR: {run_dir} is not a planned run")
    try:
        cfg = load_config(run_dir / "run" / "config.yaml")   # the run's copy, as detect.py
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")
    if args.frequency and args.frequency not in cfg.periods:
        raise SystemExit(f"STATUS ERROR: this run has no {args.frequency} periods")
    chosen = [args.frequency] if args.frequency else list(cfg.periods)
    planned = tiles_per_subregion(run_dir)
    subregions = sorted(planned)
    rows = tasks(cfg, run_dir, chosen, planned)

    if args.todo:                    # only commands, so the output can go into a file
        for row in rows:
            if row["state"] not in ("done", "running"):
                print(command(args.run, row))
        raise SystemExit(0)

    print(f"run         {run_dir}")
    print(f"subregions  {', '.join(map(str, subregions))} (those with tiles; one mark each, "
          f"in this order)")
    for freq in chosen:
        mine = [r for r in rows if r["frequency"] == freq]
        counts = Counter(r["state"] for r in mine)
        summary = ", ".join(f"{counts[s]} {s}" for s in MARKS if counts[s])
        saved, of = sum(r["saved"] for r in mine), sum(r["planned"] for r in mine)
        print(f"\n{freq}: {len(mine)} tasks: {summary}")
        print(f"{bar(saved, of)}  {saved:,}/{of:,} tiles saved")
        print("\n".join(grid(mine, freq, subregions)))
    print(f"\n{LEGEND}")

    for title, states in (("running now", ("running",)),
                          ("look at these", ("failed", "stopped", "partly done"))):
        listed = [r for r in rows if r["state"] in states]
        if listed:
            print(f"\n{title}:")
            for r in listed:
                print(f"  {r['frequency']} {r['period'].tag} subregion {r['subregion']}: "
                      f"{r['note']}")

    # ADDITION (Nov 2026): progress per task in tiles, and where the running ones run.
    shown = [r for r in rows if args.all or r["state"] not in ("done", "not started")]
    if shown:
        print("\nprogress per task (tiles saved):")
        for r in shown:
            print(f"  {r['frequency']:<9} {r['period'].tag:<8} subregion {r['subregion']:<3} "
                  f"{bar(r['saved'], r['planned'])}  {r['saved']:>6,}/{r['planned']:<6,} "
                  f"{r['state']}" + (f", {r['job']}" if r["job"] else ""))
    jobs = Counter(r["job"] for r in rows if r["job"])
    print("\nrunning on: " + ("nothing right now" if not jobs else
                              "; ".join(f"{job} ({k} task{'s' * (k > 1)})"
                                        for job, k in sorted(jobs.items()))))
