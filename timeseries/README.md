# timeseries/ — monthly-vs-quarterly pilot (phase one)

Our own code, deliberately **outside `core/`** so `git checkout upstream/main -- core/`
can never clobber it (see `GIT_PLAYBOOK.md`). Pinned against upstream/main `c473933`.

## What's here

| File | Purpose |
|---|---|
| `coverage.py` | **The cloud flag.** Per-pixel `N_CLEAR` / `N_TOTAL` counts on the same 576×576 grid as the imagery — distinguishes "no mine" from "no data", and verifies the fill=0 assumption empirically. |
| `panel.py` | Builds a per-period **chip panel**: CNN probability for *every* chip (not just positives) + coverage stats. One GeoParquet per period. |
| `run_trials.py` | Orchestrator: 12 monthly + 4 quarterly windows, resume-safe, cached, logged. |
| `analysis.py` | Statistical comparison monthly vs quarterly → `data/timeseries/analysis/REPORT.md` + figures + CSV tables. Read its docstring for the methodological cautions. |
| `explore_results.ipynb` | Thin notebook over the analysis outputs for interactive digging. |
| `run_overnight.sh` | One-command overnight launch (trials → analysis), detached via nohup. |

AOI: `data/boundaries/la_pampa_corridor.geojson` (~1,900 km², ~70 tiles,
La Pampa ASGM corridor, subset of your `madre_de_dios_mining_aoi`).

## Tonight, step by step

```bash
cd /Users/rubygrambauer/RA_Raji/AMW-mining-detector
source venv/bin/activate

# 1. smoke test (~2-3 min): 2 tiles, one month
python timeseries/run_trials.py --months 2025-07 --no-quarterly --limit_tiles 2

# 2. if that wrote data/timeseries/panels/panel_m2025-07.parquet, clear it and launch:
rm data/timeseries/panels/panel_m2025-07.parquet   # it was a 2-tile stub
caffeinate -i bash timeseries/run_overnight.sh      # caffeinate = no sleep

# 3. morning:
open data/timeseries/analysis/REPORT.md
jupyter lab timeseries/explore_results.ipynb
```

Expected duration for the full run: roughly 4–7 h (≈70 tiles × 16 periods,
cold fetches dominate; coverage fetches add ~30 %). If a period fails, rerun
`run_overnight.sh` — resume mode retries only what's missing.

## Outputs

```
data/timeseries/
  panels/panel_m2025-01.parquet …  panel_q2025Q4.parquet   # chip panels (the data)
  tiles/tile_stats_*.csv           # per-tile timing, coverage, fill0 verification
  detections/detections_*.geojson  # prob > 0.4 chips, drag into QGIS
  cache/2025/*.tif                 # imagery + coverage tiles (gitignored, reusable)
  analysis/REPORT.md, figs/, tables/
  runlog_*.log, nohup_overnight.log
```

## Panel schema (one row = one chip × one period)

`chip_id` (stable across periods) · `tile_key` · `row`,`col` (window origin in the
576-px tile) · `prob` (CNN ensemble score, continuous) · `clear_frac`
(**cloud flag**: share of pixels with ≥1 clear scene; 1 = fully observed,
0 = fully cloud-gapped) · `n_clear_mean/min` (scenes entering the median —
precision proxy) · `n_total_mean` · `geometry` (EPSG:4326) · `period`,
`cadence`, `start`, `end`.
