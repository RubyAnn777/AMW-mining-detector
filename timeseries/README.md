<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# timeseries/ — monthly vs quarterly composites

Fork work, deliberately **outside `core/`** so `git checkout upstream/main -- core/` can
never clobber it (`docs/GIT_PLAYBOOK.md`). `core/` is byte-identical to upstream — verified
for all 16 files, see `docs/CODE_WALKTHROUGH.md` §0.5.

## Run it

```bash
source ~/venvs/amw/bin/activate
cd ~/RA_Raji/AMW-mining-detector
bash timeseries/run_staged.sh 2025-01 2025-02 2025-03 q2025Q1
```

**`run_staged.sh` is the only entry point.** Nothing to export, every default correct.

```bash
bash timeseries/run_staged.sh --dry-run 2025-01     # print config, run nothing
bash timeseries/run_staged.sh --force  2025-01      # rebuild an existing panel
bash timeseries/run_staged.sh --analyse q2025Q4     # batch, then write REPORT.md
```

Period labels: months `YYYY-MM`, quarters `qYYYYQN`. Any mix, order or number.
A different year: `AMW_YEAR=2024 bash timeseries/run_staged.sh 2024-01 …`

**Do not use `caffeinate`, and do not run all 16 periods at once.** The pipeline is
resume-safe: if the Mac sleeps or you Ctrl-C, re-run the same command and it continues.
Forcing the machine awake at full load for ten hours is what overheated the laptop —
`run_overnight.sh` did exactly that and was deleted 2026-08-07.

## What's here

| file | purpose |
|---|---|
| `run_staged.sh` | **Start here.** One period per process, cooldown between, CPU caps, background priority, resume-safe. |
| `resource_limits.py` | Caps TensorFlow/BLAS threads before TF initialises. Called by `run_trials.py`. |
| `run_trials.py` | Orchestrator over monthly + quarterly windows. Invoked by `run_staged.sh`. |
| `panel.py` | Builds the per-period **chip panel**: probability for *every* chip, the six ensemble members, coverage stats, provenance. One GeoParquet per period. |
| `coverage.py` | **The cloud flag.** Per-pixel `N_CLEAR`/`N_TOTAL` on the same 576×576 grid as the imagery — distinguishes "no mine" from "no data". |
| `analysis.py` | Monthly vs quarterly comparison → `analysis/REPORT.md` + figures + CSV tables. Read its docstring for the methodological cautions. |
| `explore_results.ipynb` | Interactive digging over the analysis outputs. |

AOI: `data/boundaries/la_pampa_corridor.geojson` — ~1,900 km², **80 tiles**, 529 chips per
tile, 42,320 chips per period. La Pampa ASGM corridor, Madre de Dios.

## Protection while running (automatic)

4 TensorFlow threads, 4 EE workers, `taskpolicy -b` background priority, and a rest of 75 %
of each period's own runtime (minimum 10 min) between periods. Override with
`AMW_THREADS`, `AMW_WORKERS`, `AMW_COOLDOWN_PCT`; `AMW_THREADS=0` disables capping.
Rationale and measurements: `docs/RUN2_SETUP_AND_AUDIT.md` §3.

## Timing (measured, run 1: 80 tiles, `max_workers=8`, uncapped)

| | wall clock |
|---|---|
| monthly period | 9.9–13.8 min (median 12.0) |
| quarterly period | 27.2–29.2 min |
| 16 periods, cold cache | ~4.3 h + cooldowns |

One period once took 5.7 h from network degradation (`SSLEOFError` retries), not CPU —
there is no per-period timeout, so an unattended run can stall silently. Expect ~1.5–2×
these figures with the thread caps on.

## Outputs

```
data/timeseries_run2/
  panels/panel_m2025-01.parquet …          chip panels (the data)
  tiles/tile_stats_*.csv                   per-tile timing, coverage, fill0 verification
  detections/detections_*.geojson          prob > 0.4 chips, for QGIS
  analysis/REPORT.md, figs/, tables/
  staged_*.log, runlog_*.log
data/timeseries/cache/<year>/              imagery + coverage tiles (gitignored, reusable)
```

The cache is ~4.8 GB per year and is never evicted — deleting it costs a full re-fetch,
keeping it makes any re-run nearly free.

## Panel schema (v2) — one row per chip × period

`chip_id` (stable across periods) · `tile_key` · `row`,`col` · `prob` (ensemble mean,
continuous — **never thresholded at source**) · `p_m0…p_m5` (the six member probabilities;
`prob` is exactly their mean) · `clear_frac` (**cloud flag**: 1 = fully observed,
0 = fully cloud-gapped) · `n_clear_mean/min` · `n_total_mean` · `geometry` (EPSG:4326) ·
`period`, `cadence`, `start`, `end` · provenance: `schema_version`, `model`,
`model_sha256`, `ensemble_members`, `n_members`, `clear_threshold`, `collection`,
`tilesize`, `pad`, `chip_size`, `stride`, `git_commit`.

**Which thresholds you can still change afterwards:** the mining cutoff (`prob` is stored
continuously) and the observed-vs-missing cutoff (`clear_frac` is stored) — both free.
The **cloud threshold** (`clear_threshold=0.6`) is baked into the composite and would need
a full re-fetch into a separate cache directory.
