"""Small helpers shared by the marimo notebooks (panel_overview, explore_runs, performance,
amw_comparison/compare).

Two things live here:

* `find_runs`: the run folders next to the one given on the command line, for the
  notebooks' run picker.
* `cached`: compute a whole-run table once and keep it in <run>/summaries/, so the
  notebooks open in seconds for everyone who views them. Whoever can write to the run
  folder (the run's owner, or the summaries job) fills the cache; a viewer who can only
  read it gets the stored table, or, if there is none yet, computes it in memory and
  writes nothing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

SUMMARIES = "summaries"

# One colour per frequency, the same in every chart of every notebook.
MONTHLY, QUARTERLY = "#2a78d6", "#eb6834"


def is_run(folder: Path) -> bool:
    return (folder / "run" / "config.yaml").is_file()


def find_runs(default: str | Path | None) -> dict[str, str]:
    """{name: path} of the runs in the folder that holds `default`, newest name last.

    `default` is a run folder (from --run, or $RUN). Its siblings that are runs too are
    offered as well, so a viewer can switch between runs without typing a path.
    """
    if not default:
        return {}
    default = Path(default).expanduser().resolve()
    runs = {default.name: str(default)} if is_run(default) else {}
    try:
        siblings = [p for p in default.parent.iterdir() if p.is_dir() and is_run(p)]
    except OSError:                     # the parent folder is not readable
        siblings = []
    for p in siblings:
        runs.setdefault(p.name, str(p))
    return dict(sorted(runs.items()))


def newest_change(paths: Iterable[Path]) -> float:
    """The latest modification time among `paths` (0 if none exists)."""
    times = []
    for p in paths:
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(times, default=0.0)


def is_cached(run: Path, name: str, inputs: Iterable[Path]) -> bool:
    """True if `cached` would return a stored result for `name` without computing."""
    path = Path(run) / SUMMARIES / f"{name}.pkl"
    return path.is_file() and path.stat().st_mtime >= newest_change(inputs)


def can_write(run: Path) -> bool:
    """True if this user can store summaries in the run (its owner can; a viewer can't)."""
    folder = Path(run) / SUMMARIES
    return os.access(folder if folder.is_dir() else Path(run), os.W_OK)


def cached(run: Path, name: str, compute: Callable[[], object], inputs: Iterable[Path]) -> object:
    """compute(), stored as <run>/summaries/<name>.pkl and reused while it is up to date.

    The stored result is reused when it is newer than every file in `inputs` (for example
    the panel's _record.json files: a newly assembled period makes it stale). A user
    without write access to the run gets the result without anything being written.
    """
    path = Path(run) / SUMMARIES / f"{name}.pkl"
    inputs = list(inputs)
    try:
        if path.is_file() and path.stat().st_mtime >= newest_change(inputs):
            return pd.read_pickle(path)
    except Exception:                   # unreadable or from an incompatible pandas: recompute
        pass
    value = compute()
    tmp = path.with_name(f"_{path.name}.{os.getpid()}")
    try:
        path.parent.mkdir(exist_ok=True)
        pd.to_pickle(value, tmp)
        os.replace(tmp, path)           # appears complete or not at all
    except OSError:                     # read-only for this user: keep it in memory only
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return value


# A chart sends every row it draws to the browser, and marimo refuses outputs above a few
# megabytes (8 MB). A full run has hundreds of thousands of chips per quarter, so point charts
# draw a random sample of at most this many rows and say so in their title.
MAX_POINTS = 10_000


def thin(df: pd.DataFrame, n: int = MAX_POINTS, every: str | None = None) -> tuple[pd.DataFrame, str]:
    """At most `n` rows of `df` for a chart, and a note saying how many were left out.

    Without `every`, a random sample (the same one each time). With `every`, a column name:
    rows are kept at even steps within each value of that column, in their order, so a
    line per task stays a line.
    """
    if len(df) <= n:
        return df, ""
    if every is None:
        out = df.sample(n, random_state=0)
    else:
        step = -(-len(df) // n)                                   # ceiling division
        out = df[df.groupby(every).cumcount() % step == 0]
    return out, f"showing {len(out):,} of {len(df):,}, picked at random" if every is None else \
        f"showing one in {step} of {len(df):,} points"
