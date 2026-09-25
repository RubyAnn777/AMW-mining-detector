#!/usr/bin/env python3
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
run_trials.py — overnight orchestrator: 12 monthly + 4 quarterly chip panels
for one AOI and one calendar year (default: La Pampa corridor, 2025).

Resume-safe: a period whose panel parquet already exists is skipped, so if the
run dies at 3 a.m. you just start it again and it continues where it stopped.
Imagery/coverage tiles are additionally cached per (tile, window) on disk, so
even a partially-finished period restarts nearly warm.

Typical use (from repo root, venv active, Earth Engine authenticated):

    export EE_PROJECT=<your-gcp-project>          # or set in .env, see run_staged.sh
    python timeseries/run_trials.py               # full 2025 trial, both cadences

Smoke test before bed (2 tiles, 1 month, ~2 minutes):

    python timeseries/run_trials.py --months 2025-07 --no-quarterly --limit_tiles 2

Outputs land under data/timeseries/:
    panels/panel_m2025-01.parquet ... panel_q2025Q4.parquet   (chip panels)
    tiles/tile_stats_*.csv                                    (timing + coverage + fill0 check)
    detections/detections_*.geojson                           (positives, for QGIS)
    runlog_<timestamp>.log
"""
# >> FILE OVERVIEW: fork-original (not in upstream). Entry point for the time-series
# >> panels.
# >> Builds the list of month / quarter windows for one year and calls
# >> panel.py:build_period_panel for each, skipping any already done (resume-safe).
# >> Usually launched once per period by run_staged.sh (--months <label>).
# >> Outputs feed analysis.py.
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# paths: make upstream core/ and this folder importable regardless of cwd     #
# --------------------------------------------------------------------------- #
# >> Repo root; core/ goes on sys.path so upstream modules import by bare name (import
# >> gee).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "core"))       # upstream modules (gee, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # coverage, panel

# >> ORDER MATTERS: the thread caps must be set before TensorFlow is imported (below).
# CPU thread caps MUST be applied before TensorFlow/numpy are imported, i.e. before
# `import panel` pulls in inference_engine -> tensorflow. See resource_limits.py and
# docs/RUN2_SETUP_AND_AUDIT.md §3. Budget: AMW_THREADS env var, default 4, 0 = uncapped.
from resource_limits import apply_resource_limits, describe  # noqa: E402

# >> Sets OMP/BLAS env vars and TF/torch thread pools (AMW_THREADS, default 4).
# >> Gotcha: logging is not configured yet (basicConfig is in main), so its INFO lines are
# >> dropped; main logs describe() later to record the caps.
apply_resource_limits()

# >> This import pulls in TensorFlow (via inference_engine), hence it comes after the
# >> caps.
from panel import PeriodSpec, build_period_panel  # noqa: E402


# >> What: the 12 calendar-month windows of a year.
# >> Out: list of PeriodSpec, labels 'm2025-01'..'m2025-12', cadence 'monthly'.
# >> End date EXCLUSIVE: Jan = 2025-01-01 -> 2025-02-01; Dec ends 2026-01-01. So the last
# >> day is included, unlike upstream periods.py, whose 03-31 / 12-31 end dates drop that
# >> day.
# >> Used by main.
def month_specs(year: int) -> list[PeriodSpec]:
    """12 monthly windows; end date exclusive (ee.filterDate convention)."""
    specs = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01"
        # >> December rolls over to 1 January of the next year.
        end = f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01"
        specs.append(PeriodSpec(f"m{year}-{m:02d}", start, end, "monthly"))
    return specs


# >> What: 4 quarter windows: Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec. (start
# >> month, end month) pairs; 13 means "1 January of next year". End exclusive.
# >> Out: labels 'q2025Q1'..'q2025Q4', cadence 'quarterly'. Used by main.
def quarter_specs(year: int) -> list[PeriodSpec]:
    """4 quarterly windows: Q1 = Jan1..Apr1(excl), etc."""
    starts = [(1, 4), (4, 7), (7, 10), (10, 13)]
    specs = []
    for qi, (m0, m1) in enumerate(starts, start=1):
        start = f"{year}-{m0:02d}-01"
        end = f"{year + 1}-01-01" if m1 == 13 else f"{year}-{m1:02d}-01"
        specs.append(PeriodSpec(f"q{year}Q{qi}", start, end, "quarterly"))
    return specs


# >> What: command-line entry; runs each selected period, logs, returns an exit code.
# >> Defaults: La Pampa AOI, the production 6-CNN ensemble, year 2025, out
# >> data/timeseries.
# >> Writes under out_root: panels/, tiles/, detections/ (via panel.py), runlog_<ts>.log.
# >> Exit 0 = all done, 1 = some period failed, 2 = no periods selected.
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", type=Path,
                    default=REPO_ROOT / "data/boundaries/la_pampa_corridor.geojson")
    ap.add_argument("--model", type=Path,
                    default=REPO_ROOT / "models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out_root", type=Path, default=REPO_ROOT / "data/timeseries")
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="default: <out_root>/cache/<year>")
    # >> Only the positives GeoJSON uses this; 0.4 = the basin inference floor used
    # >> upstream.
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="only affects the convenience detections geojson; the "
                         "panel always stores ALL probabilities")
    # >> Parallel EE downloads per batch, and the batch size, in panel.py.
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--limit_tiles", type=int, default=None,
                    help="smoke-test: only the first N tiles")
    ap.add_argument("--months", type=str, default=None,
                    help="comma list like 2025-07,2025-08 to run a subset")
    ap.add_argument("--no-monthly", action="store_true")
    ap.add_argument("--no-quarterly", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild panels even if the parquet exists")
    args = ap.parse_args()

    # >> Cache default: <out_root>/cache/<year>; panels always go to <out_root>/panels.
    cache_dir = args.cache_dir or (args.out_root / "cache" / str(args.year))
    panels_dir = args.out_root / "panels"
    args.out_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # >> Log to the console and to FILE runlog_<YYYYmmdd_HHMMSS>.log in out_root.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.out_root / f"runlog_{stamp}.log")])
    log = logging.getLogger("run_trials")

    # ------------------------------------------------------------- period list
    specs: list[PeriodSpec] = []
    # >> Processing order: the 12 months first, then the 4 quarters.
    if not args.no_monthly:
        specs += month_specs(args.year)
    if not args.no_quarterly:
        specs += quarter_specs(args.year)
    # >> --months filter: 'm2025-07' matches '2025-07' (leading m stripped); quarters
    # >> match only by full label, e.g. 'q2025Q1'. So '--months 2025-07' alone drops all
    # >> quarters.
    if args.months:
        wanted = {s.strip() for s in args.months.split(",")}
        specs = [s for s in specs
                 if s.label.lstrip("m") in wanted or s.label in wanted]
    if not specs:
        log.error("No periods selected."); return 2

    log.info("AOI=%s  model=%s  year=%d  periods=%s", args.region.name,
             args.model.name, args.year, [s.label for s in specs])
    # Record the effective CPU caps in the runlog so a slow run can be attributed
    # to throttling rather than guessed at afterwards.
    log.info("resource limits: %s | max_workers=%d", describe(), args.max_workers)

    # ------------------------------------------------------------------- run
    failures = []
    # >> Serial loop over periods: one period fully finishes before the next starts.
    for spec in specs:
        panel_path = panels_dir / f"panel_{spec.label}.parquet"
        # >> Resume-safe skip: an existing parquet means done (it is only written at
        # >> period end).
        if panel_path.exists() and not args.force:
            log.info("[%s] panel exists, skipping (resume mode)", spec.label)
            continue
        t0 = time.perf_counter()
        try:
            # >> Heavy work: fetch + score + write this period
            # >> (panel.py:build_period_panel).
            build_period_panel(
                spec, region_path=args.region, model_path=args.model,
                out_dir=panels_dir, cache_dir=cache_dir,
                pred_threshold=args.threshold, max_workers=args.max_workers,
                limit_tiles=args.limit_tiles)
            log.info("[%s] finished in %.1f min", spec.label,
                     (time.perf_counter() - t0) / 60)
        except Exception:
            # one bad period must not kill the night — log and continue
            log.exception("[%s] PERIOD FAILED after %.1f min", spec.label,
                          (time.perf_counter() - t0) / 60)
            failures.append(spec.label)

    # >> Non-zero exit tells run_staged.sh this period FAILED.
    if failures:
        log.error("Done with FAILURES: %s (rerun me — resume mode will retry "
                  "only these)", failures)
        return 1
    log.info("All periods done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
