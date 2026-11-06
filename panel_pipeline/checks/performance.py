"""How fast a run went and what limited it: the numbers of README B4, read off any run.

    python panel_pipeline/checks/performance.py                 # on Goethe: $RUN, as set at login
    python panel_pipeline/checks/performance.py --run tests_output/<run>      # a laptop copy

Reads every task folder's `_log.txt` (the progress line detect.py writes every few
minutes) and `_status.json` (its sessions), plus the jobs: on the cluster it asks
`sacct` itself and saves the answer as <run>/jobs.txt; elsewhere it reads that file if
the copy of the run has it. (By hand, on the cluster:
`sacct --starttime <first day> -X -P --format=JobID,JobName,NodeList,Start,End,Elapsed,State > $RUN/jobs.txt`.)

Prints one line per task (tiles, wall time, tiles/min, download median, what was busy),
one per job (node, hours, core-hours, tiles scored in it) and the run's totals. The
notebook next to it, `performance.ipynb`, shows the same numbers over time and what the
full plan would cost at the measured pace. Reads only; works on a laptop copy of the run
too (`rsync` the `detect/` folders, README B6: the log and status files are small).

The progress line (detect.describe_window), as it looks in _log.txt:
    2026-10-28T12:56:15+00:00  2025-01  try 1: 408/53,084 tiles saved (+57, 0 failed)
    | 28 tiles/min, download median 6 s (slowest 14 s), workers busy 96%, scoring busy
    92%, 0 retries | 2 downloading, 0 waiting for a worker, 3 waiting to be scored,
    oldest asked for 6 s ago | about 31.4 h left in this try
"workers busy" near 100% while "scoring busy" is well below: Earth Engine (or the node's
link to it) is the limit. "scoring busy" near 100%: the node's TensorFlow scoring is.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TASK_COLUMNS = ["frequency", "period", "subregion"]

# One progress line: the counts, then the window (two forms), then the queue.
_HEAD = re.compile(
    r"^(?P<time>\S+)\s+(?P<period>\S+)\s+try (?P<try>\d+): (?P<saved>[\d,]+)/(?P<planned>[\d,]+) "
    r"tiles saved \(\+(?P<added>\d+), (?P<failed>\d+) failed\) \| (?P<window>.*?) \| "
    r"(?P<downloading>\d+) downloading, (?P<waiting_worker>\d+) waiting for a worker, "
    r"(?P<waiting_score>\d+) waiting to be scored, oldest asked for (?P<oldest_s>\d+) s ago")
_WINDOW = re.compile(
    r"^(?P<rate>\d+) tiles/min, download median (?P<dl_median_s>\d+) s \(slowest "
    r"(?P<dl_max_s>\d+) s\), workers busy (?P<workers_busy>\d+)%, scoring\s*busy "
    r"(?P<scoring_busy>\d+)%, (?:EE busy )?(?P<retries>\d+)(?: retries|x)$")   # "EE busy 3x" since Nov 2026
_EMPTY_WINDOW = re.compile(r"^no tile finished in [\d.]+ min, (?:EE busy )?(?P<retries>\d+)(?: retries|x)$")
_SESSION_START = re.compile(
    r"^(?P<time>\S+)\s+(?P<period>\S+)\s+(?:resuming: (?P<saved>[\d,]+)/[\d,]+ tiles already saved"
    r".*?|starting:) (?P<to_score>[\d,]+) tiles to score, (?P<workers>\d+) workers")

NUMBER_COLUMNS = ["try", "saved", "planned", "added", "failed", "rate", "dl_median_s",
                  "dl_max_s", "workers_busy", "scoring_busy", "retries", "downloading",
                  "waiting_worker", "waiting_score", "oldest_s"]


def _int(text: str) -> int:
    return int(text.replace(",", ""))


# ------------------------------------------------------------ the task folders

def task_folders(run_dir: Path) -> list[Path]:
    """Every task folder that detect.py has written to, period first."""
    return sorted(p for p in run_dir.glob("detect/*/period=*/subregion=*")
                  if (p / "_status.json").is_file() or (p / "_log.txt").is_file())


def task_of(folder: Path) -> dict:
    """frequency, period and subregion from a task folder's path."""
    return {"frequency": folder.parent.parent.name,
            "period": folder.parent.name.split("=", 1)[1],
            "subregion": int(folder.name.split("=", 1)[1])}


# ------------------------------------------------------------ _log.txt

def parse_progress(text: str) -> list[dict]:
    """The progress lines of one _log.txt, as dicts; other lines are skipped."""
    rows = []
    for line in text.splitlines():
        head = _HEAD.match(line.strip())
        if not head:
            continue
        row = head.groupdict()
        window = _WINDOW.match(row.pop("window"))
        if window:
            row |= window.groupdict()
        else:
            empty = _EMPTY_WINDOW.match(head["window"])
            if not empty:
                continue
            row |= {"rate": "0", "dl_median_s": None, "dl_max_s": None, "workers_busy": None,
                    "scoring_busy": None, "retries": empty["retries"]}
        rows.append({k: (_int(v) if k in NUMBER_COLUMNS and v is not None else v)
                     for k, v in row.items()})
    return rows


def parse_session_starts(text: str) -> list[dict]:
    """The 'starting:' and 'resuming:' lines: when a session began, from how many tiles."""
    rows = []
    for line in text.splitlines():
        m = _SESSION_START.match(line.strip())
        if m:
            rows.append({"time": m["time"], "saved_before": _int(m["saved"] or "0"),
                         "to_score": _int(m["to_score"]), "workers": int(m["workers"])})
    return rows


def progress(run_dir: Path) -> pd.DataFrame:
    """Every progress line of every task, with the task's columns and a UTC time."""
    frames = []
    for folder in task_folders(run_dir):
        log = folder / "_log.txt"
        if not log.is_file():
            continue
        rows = parse_progress(log.read_text())
        if rows:
            frames.append(pd.DataFrame(rows).assign(**task_of(folder)))
    if not frames:
        return pd.DataFrame(columns=TASK_COLUMNS + ["time"] + NUMBER_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["task"] = out["period"] + " s" + out["subregion"].astype(str)
    return out.sort_values(["frequency", "period", "subregion", "time"]).reset_index(drop=True)


# ------------------------------------------------------------ _status.json

def sessions(run_dir: Path, lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per detect.py session of every task: when, with what settings, how many tiles.

    `lines` (from progress()) gives each session the tiles it scored and its line
    statistics; without it, only what _status.json knows.
    """
    rows = []
    for folder in task_folders(run_dir):
        path = folder / "_status.json"
        if not path.is_file():
            continue
        status = json.loads(path.read_text())
        starts = parse_session_starts((folder / "_log.txt").read_text()) \
            if (folder / "_log.txt").is_file() else []
        for i, s in enumerate(status.get("sessions", [])):
            end = s.get("finished_utc") or s.get("stopped_utc")
            start = next((st for st in starts if st["time"] >= s["started_utc"]
                          and (end is None or st["time"] <= end)), None)
            rows.append(task_of(folder) | {
                "session": i + 1, "started": s["started_utc"], "ended": end,
                "how": "finished" if s.get("finished_utc") else
                       "stopped" if s.get("stopped_utc") else "lost",
                "workers": s.get("max_workers"), "limit": s.get("limit"),
                "saved_before": start["saved_before"] if start else None,
                "tiles_planned": status["tiles_planned"], "tiles_saved": status["tiles_saved"],
                "complete": status["complete"], "tiles_failed": len(status["tiles_failed"])})
    columns = TASK_COLUMNS + ["session", "started", "ended", "how", "workers", "limit",
                              "saved_before", "tiles_planned", "tiles_saved", "complete",
                              "tiles_failed"]
    out = pd.DataFrame(rows, columns=columns)
    out["started"] = pd.to_datetime(out["started"], utc=True)
    out["ended"] = pd.to_datetime(out["ended"], utc=True)
    out["hours"] = (out["ended"] - out["started"]).dt.total_seconds() / 3600
    out["task"] = out["period"] + " s" + out["subregion"].astype(str)
    out["tiles"] = np.nan                        # scored in this session, from the lines
    out["rate"] = np.nan                         # tiles/min, median of the lines
    out["dl_median_s"] = out["workers_busy"] = out["scoring_busy"] = np.nan
    out["retries"] = np.nan
    if lines is None or lines.empty:
        return out
    for i, s in out.iterrows():
        mine = lines[(lines["frequency"] == s["frequency"]) & (lines["period"] == s["period"])
                     & (lines["subregion"] == s["subregion"]) & (lines["time"] >= s["started"])
                     & ((lines["time"] <= s["ended"]) if pd.notna(s["ended"]) else True)]
        if mine.empty:
            continue
        before = s["saved_before"] if pd.notna(s["saved_before"]) \
            else mine["saved"].iloc[0] - mine["added"].iloc[0]
        out.loc[i, "tiles"] = mine["saved"].max() - before
        busy = mine[mine["rate"] > 0]
        out.loc[i, "rate"] = busy["rate"].median() if len(busy) else 0
        out.loc[i, "dl_median_s"] = busy["dl_median_s"].median()
        out.loc[i, "workers_busy"] = busy["workers_busy"].median()
        out.loc[i, "scoring_busy"] = busy["scoring_busy"].median()
        out.loc[i, "retries"] = mine["retries"].sum()
    return out


# ------------------------------------------------------------ sacct

def _hours(elapsed: str) -> float:
    """Slurm's D-HH:MM:SS or HH:MM:SS or MM:SS, in hours."""
    days, _, clock = elapsed.rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return int(days or 0) * 24 + h + m / 60 + s / 3600


def jobs(path: Path, cluster_tz: str = "Europe/Berlin") -> pd.DataFrame:
    """A saved `sacct -X` listing (-P pipe-separated, or the plain table), one row per job.

    sacct prints the cluster's local time without a zone; `cluster_tz` turns it into UTC
    so jobs and sessions can be matched. Wanted columns: JobID, NodeList, Start, End,
    Elapsed, State (JobName is nice to have).
    """
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip() and not set(ln.strip()) <= {"-", " "}]
    if "|" in lines[0]:
        rows = [ln.split("|") for ln in lines]
    else:
        rows = [ln.split() for ln in lines]
    header, body = rows[0], rows[1:]
    out = pd.DataFrame([dict(zip(header, r)) for r in body if len(r) == len(header)])
    out = out.rename(columns={"JobID": "job", "JobName": "name", "NodeList": "node",
                              "Start": "start", "End": "end", "Elapsed": "elapsed",
                              "State": "state"})
    for col in ("start", "end"):
        if col in out:
            out[col] = (pd.to_datetime(out[col].replace({"Unknown": None, "None": None}))
                        .dt.tz_localize(cluster_tz, ambiguous="NaT", nonexistent="shift_forward")
                        .dt.tz_convert("UTC"))
    if "elapsed" in out:
        out["hours"] = out["elapsed"].map(_hours)
    return out


SACCT_FORMAT = "JobID,JobName,NodeList,Start,End,Elapsed,State"


def find_jobs(run_dir: Path, sess: pd.DataFrame, path: Path | None = None,
              cluster_tz: str = "Europe/Berlin") -> tuple[pd.DataFrame | None, str]:
    """The jobs listing, from a saved file or from sacct itself; and a line saying which.

    Looks for `path` (default <run>/jobs.txt). If there is none and this machine has
    `sacct` (a Slurm cluster), asks it for the user's jobs since the run's first session
    and saves the answer as <run>/jobs.txt, so a laptop copy of the run carries it too.
    """
    path = Path(path) if path else run_dir / "jobs.txt"
    if path.is_file():
        return jobs(path, cluster_tz), f"jobs listing: {path}"
    if not shutil.which("sacct"):
        return None, (f"jobs listing: none ({path} not found, and no sacct here: save one on the "
                      f"cluster, see the module docstring)")
    since = (sess["started"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if len(sess) \
        else "2026-01-01"
    result = subprocess.run(["sacct", "--starttime", since, "-X", "-P", "--format", SACCT_FORMAT],
                            capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None, f"jobs listing: sacct failed ({result.stderr.strip()[:200]})"
    path.write_text(result.stdout)
    return jobs(path, cluster_tz), f"jobs listing: asked sacct (since {since}) and saved it as {path}"


def assign_jobs(sess: pd.DataFrame, job_table: pd.DataFrame | None,
                slack_minutes: float = 3) -> pd.DataFrame:
    """Give every session the job (and node) that was running when it started."""
    out = sess.copy()
    out["job"] = None
    out["node"] = None
    if job_table is None or job_table.empty or "start" not in job_table:
        return out
    slack = pd.Timedelta(minutes=slack_minutes)
    ends = job_table["end"] if "end" in job_table else pd.Series(pd.NaT, index=job_table.index)
    ends = ends.fillna(pd.Timestamp("2262-01-01", tz="UTC"))     # still running: no end yet
    for i, s in out.iterrows():
        hit = job_table[(job_table["start"] - slack <= s["started"]) & (ends + slack >= s["started"])]
        if len(hit):
            out.loc[i, "job"] = hit["job"].iloc[0]
            out.loc[i, "node"] = hit["node"].iloc[0]
    return out


# ------------------------------------------------------------ summaries

def limited_by(workers_busy: float, scoring_busy: float) -> str:
    """What the busy figures say held a task back."""
    if pd.isna(workers_busy) or pd.isna(scoring_busy):
        return "?"
    if scoring_busy >= 80 and workers_busy >= 80:
        return "both (balanced)"
    if scoring_busy >= 80:
        return "scoring (the node)"
    if workers_busy >= 80:
        return "downloads (Earth Engine or the link)"
    return "neither (short, or stopping)"


def task_summary(sess: pd.DataFrame) -> pd.DataFrame:
    """One line per task from its sessions: tiles, hours, speed, what limited it."""
    if sess.empty:
        return pd.DataFrame()
    if "node" not in sess:                       # assign_jobs() was not run
        sess = sess.assign(node=None)
    g = sess.groupby(TASK_COLUMNS, sort=True)
    out = g.agg(tiles_planned=("tiles_planned", "last"), tiles_saved=("tiles_saved", "last"),
                complete=("complete", "last"), failed=("tiles_failed", "last"),
                sessions=("session", "max"), hours=("hours", "sum"),
                tiles_scored=("tiles", "sum"), rate=("rate", "median"),
                dl_median_s=("dl_median_s", "median"), workers_busy=("workers_busy", "median"),
                scoring_busy=("scoring_busy", "median"), retries=("retries", "sum"),
                workers=("workers", "last"), limit=("limit", "last"),
                nodes=("node", lambda c: ", ".join(sorted({str(n) for n in c if n})))
                ).reset_index()
    out["tiles_per_min"] = out["tiles_scored"] / (out["hours"] * 60)
    out["limited_by"] = [limited_by(w, s) for w, s in zip(out["workers_busy"], out["scoring_busy"])]
    return out


def job_summary(sess: pd.DataFrame, job_table: pd.DataFrame | None,
                cores_per_node: int = 40) -> pd.DataFrame:
    """One line per job: node, hours, core-hours, the tiles its sessions scored."""
    if job_table is None or job_table.empty:
        return pd.DataFrame()
    out = job_table.copy()
    out["core_hours"] = out["hours"] * cores_per_node
    scored = sess.groupby("job")["tiles"].sum() if "job" in sess else pd.Series(dtype=float)
    tasks = sess.groupby("job")["task"].agg(lambda c: ", ".join(c.unique())) if "job" in sess \
        else pd.Series(dtype=str)
    out["tiles"] = out["job"].map(scored).fillna(0)
    out["tasks"] = out["job"].map(tasks).fillna("")
    out["tiles_per_core_hour"] = out["tiles"] / out["core_hours"].replace(0, np.nan)
    columns = [c for c in ("job", "name", "node", "start", "end", "hours", "core_hours", "state",
                           "tiles", "tiles_per_core_hour", "tasks") if c in out]
    return out[columns]


def totals(tasks: pd.DataFrame, job_table: pd.DataFrame | None, cores_per_node: int = 40) -> dict:
    """The run in a few numbers; core-hours only when a jobs listing was given."""
    out = {"tasks": len(tasks), "tile_periods_saved": int(tasks["tiles_saved"].sum()) if len(tasks) else 0,
           "task_hours": float(tasks["hours"].sum()) if len(tasks) else 0.0}
    if len(tasks):
        busy = tasks[tasks["tiles_per_min"] > 0]
        out["tiles_per_min_per_task_median"] = float(busy["tiles_per_min"].median()) if len(busy) else 0.0
        out["dl_median_s"] = float(tasks["dl_median_s"].median())
    if job_table is not None and not job_table.empty and "hours" in job_table:
        detect = job_table[job_table["tiles"] > 0] if "tiles" in job_table else job_table
        core_hours = float(detect["hours"].sum() * cores_per_node)
        out["jobs"] = len(detect)
        out["core_hours"] = core_hours
        scored = float(detect["tiles"].sum()) if "tiles" in detect else out["tile_periods_saved"]
        out["tile_periods_per_core_hour"] = scored / core_hours if core_hours else float("nan")
        out["tile_periods_per_node_day"] = out["tile_periods_per_core_hour"] * cores_per_node * 24
    return out


def projection(n_tiles: int, n_periods: dict[str, int], per_core_hour: float,
               eecu_per_tile: dict[str, float], cores_per_node: int = 40, nodes: int = 3,
               quota_eecu_hours: float = 100_000, allocation_core_hours: float = 200_000) -> pd.DataFrame:
    """What each scope of the plan costs at the measured pace (README B4's table, live)."""
    rows = []
    scopes = {f"{freq} only": {freq: n} for freq, n in n_periods.items()}
    if len(n_periods) > 1:
        scopes["both"] = dict(n_periods)
    for name, parts in scopes.items():
        tile_periods = sum(n_tiles * n for n in parts.values())
        core_hours = tile_periods / per_core_hour
        eecu_hours = sum(n_tiles * n * eecu_per_tile.get(freq, float("nan")) for freq, n in parts.items()) / 3600
        rows.append({"scope": name, "tile_periods": tile_periods, "core_hours": core_hours,
                     "share_of_allocation": core_hours / allocation_core_hours,
                     "node_days": core_hours / (cores_per_node * 24),
                     f"days_on_{nodes}_nodes": core_hours / (cores_per_node * 24 * nodes),
                     "eecu_hours": eecu_hours, "months_of_quota": eecu_hours / quota_eecu_hours})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ command line

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="How fast a run went, and what limited it.")
    ap.add_argument("--run", default=os.environ.get("RUN"),
                    help="a run folder, or a laptop copy of its detect/ (default: $RUN)")
    ap.add_argument("--jobs", help="a saved `sacct -X -P ...` listing (default: <run>/jobs.txt, "
                                   "or sacct itself on the cluster; see the module docstring)")
    ap.add_argument("--cluster-tz", default="Europe/Berlin", help="sacct's time zone")
    ap.add_argument("--cores", type=int, default=40, help="cores per node (core-hours)")
    args = ap.parse_args(argv)
    if not args.run:
        ap.error("--run is needed (or set RUN, as the login lines of README B3 do)")

    run_dir = Path(args.run).expanduser().resolve()
    lines = progress(run_dir)
    sess = sessions(run_dir, lines)
    job_table, note = find_jobs(run_dir, sess, args.jobs, args.cluster_tz)
    sess = assign_jobs(sess, job_table)
    tasks = task_summary(sess)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 1)
    print(f"run  {run_dir}: {len(lines):,} progress lines, {len(sess)} sessions, {len(tasks)} tasks")
    print(note, "\n")
    if len(tasks):
        show = tasks[TASK_COLUMNS + ["tiles_saved", "tiles_planned", "failed", "sessions", "hours",
                                     "tiles_per_min", "dl_median_s", "workers_busy", "scoring_busy",
                                     "retries", "workers", "nodes", "limited_by"]]
        print(show.to_string(index=False), "\n")
    job_table = job_summary(sess, job_table, args.cores)
    if len(job_table):
        print(job_table.drop(columns=["tasks"]).to_string(index=False), "\n")
    for key, value in totals(tasks, job_table, args.cores).items():
        print(f"{key:32} {value:,.1f}" if isinstance(value, float) else f"{key:32} {value:,}")


if __name__ == "__main__":
    main()
