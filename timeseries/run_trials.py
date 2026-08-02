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

    export EE_PROJECT=<your-gcp-project>          # or set in .env, see run_overnight.sh
    python timeseries/run_trials.py               # full 2025 trial, both cadences

Smoke test before bed (2 tiles, 1 month, ~2 minutes):

    python timeseries/run_trials.py --months 2025-07 --no-quarterly --limit_tiles 2

Outputs land under data/timeseries/:
    panels/panel_m2025-01.parquet ... panel_q2025Q4.parquet   (chip panels)
    tiles/tile_stats_*.csv                                    (timing + coverage + fill0 check)
    detections/detections_*.geojson                           (positives, for QGIS)
    runlog_<timestamp>.log
"""
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
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "core"))       # upstream modules (gee, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # coverage, panel

from panel import PeriodSpec, build_period_panel  # noqa: E402


def month_specs(year: int) -> list[PeriodSpec]:
    """12 monthly windows; end date exclusive (ee.filterDate convention)."""
    specs = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01"
        end = f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01"
        specs.append(PeriodSpec(f"m{year}-{m:02d}", start, end, "monthly"))
    return specs


def quarter_specs(year: int) -> list[PeriodSpec]:
    """4 quarterly windows: Q1 = Jan1..Apr1(excl), etc."""
    starts = [(1, 4), (4, 7), (7, 10), (10, 13)]
    specs = []
    for qi, (m0, m1) in enumerate(starts, start=1):
        start = f"{year}-{m0:02d}-01"
        end = f"{year + 1}-01-01" if m1 == 13 else f"{year}-{m1:02d}-01"
        specs.append(PeriodSpec(f"q{year}Q{qi}", start, end, "quarterly"))
    return specs


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
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="only affects the convenience detections geojson; the "
                         "panel always stores ALL probabilities")
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

    cache_dir = args.cache_dir or (args.out_root / "cache" / str(args.year))
    panels_dir = args.out_root / "panels"
    args.out_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.out_root / f"runlog_{stamp}.log")])
    log = logging.getLogger("run_trials")

    # ------------------------------------------------------------- period list
    specs: list[PeriodSpec] = []
    if not args.no_monthly:
        specs += month_specs(args.year)
    if not args.no_quarterly:
        specs += quarter_specs(args.year)
    if args.months:
        wanted = {s.strip() for s in args.months.split(",")}
        specs = [s for s in specs
                 if s.label.lstrip("m") in wanted or s.label in wanted]
    if not specs:
        log.error("No periods selected."); return 2

    log.info("AOI=%s  model=%s  year=%d  periods=%s", args.region.name,
             args.model.name, args.year, [s.label for s in specs])

    # ------------------------------------------------------------------- run
    failures = []
    for spec in specs:
        panel_path = panels_dir / f"panel_{spec.label}.parquet"
        if panel_path.exists() and not args.force:
            log.info("[%s] panel exists, skipping (resume mode)", spec.label)
            continue
        t0 = time.perf_counter()
        try:
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

    if failures:
        log.error("Done with FAILURES: %s (rerun me — resume mode will retry "
                  "only these)", failures)
        return 1
    log.info("All periods done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
