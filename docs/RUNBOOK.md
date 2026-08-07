<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# RUNBOOK — set up, run, don't break things

The only doc you need day to day. Evidence and audit findings live in
`RUN2_SETUP_AND_AUDIT.md`; the code tour lives in `CODE_WALKTHROUGH.md`.

---

## 1. Run

```bash
source ~/venvs/amw/bin/activate
cd ~/RA_Raji/AMW-mining-detector
bash timeseries/run_staged.sh 2025-01 2025-02 2025-03 q2025Q1
```

`run_staged.sh` is the only entry point. Nothing to export; every default is correct.

| flag | effect |
|---|---|
| `--dry-run` | print the config and exit |
| `--force` | rebuild panels that already exist |
| `--analyse` | write `REPORT.md` after the batch |

Period labels: months `YYYY-MM`, quarters `qYYYYQN`. Any mix, order or count.
Another year: `AMW_YEAR=2024 bash timeseries/run_staged.sh 2024-01 …`

**Tuning** (rarely needed):

```bash
AMW_WORKERS=8        # Earth Engine concurrency + batch size (default 4)
AMW_THREADS=2        # TensorFlow/BLAS threads (default 4; 0 = uncapped)
AMW_COOLDOWN_PCT=100 # rest time as % of the period's runtime (default 75)
```

**Analysis** — pools *every* panel in the directory, so run it only when all periods you
want compared exist:

```bash
python timeseries/analysis.py \
  --panels_dir data/timeseries_run2/panels \
  --tiles_dir  data/timeseries_run2/tiles \
  --out_dir    data/timeseries_run2/analysis
```

**Where things land**

| what | path |
|---|---|
| panels, tile stats, detections, logs | `data/timeseries_run2/` |
| tile cache (~4.8 GB/year, gitignored, reusable) | `data/timeseries/cache/<year>/` |
| run 1's completed 2025 outputs | `data/timeseries/` |
| the venv | `~/venvs/amw` (outside the repo, deliberately) |

**Resume:** re-running the same command is always safe and cheap. Finished periods are
skipped; cached tiles are reused. Ctrl-C during a cooldown loses nothing; mid-period it
loses only that period's parquet, and the tiles it fetched stay cached.

---

## 2. Set up (new machine)

```bash
# 1. a Python 3.11 interpreter — HARD requirement, tensorflow==2.14.0 has no wheels above cp311
UVPY=$(uv python find 3.11)          # or $(brew --prefix python@3.11)/bin/python3.11
"$UVPY" -VV                          # must print 3.11.x

# 2. venv OUTSIDE the repo
mkdir -p ~/venvs && "$UVPY" -m venv ~/venvs/amw
source ~/venvs/amw/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/RA_Raji/AMW-mining-detector/requirements-lock_2026-07-17.txt

# 3. Earth Engine auth — localhost mode, NOT the default
python - <<'PY'
import ee
ee.Authenticate(auth_mode='localhost',
    scopes=['https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform'], force=True)
PY

# 4. verification gate — run this every session before anything else
python -VV && which python
python -c "import ee, geopandas, tensorflow, numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

Expect `3.11.15`, a path under `~/venvs/amw/bin/`, and `1.26.4 3.0.3`.

End-to-end check (hits Earth Engine for real):

```bash
cd ~/RA_Raji/AMW-mining-detector
python - <<'PY'
import sys; sys.path.insert(0, "core")
import gee, tile_utils, geopandas as gpd
cfg = gee.DataConfig()
ex  = gee.GEE_Data_Extractor("2025-07-01", "2025-08-01", cfg)
region = gpd.read_file("data/boundaries/la_pampa_corridor.geojson").union_all()
tiles  = tile_utils.create_tiles(region, cfg.tilesize, cfg.pad)
print("tiles:", len(tiles))                       # expect 80
print("shape:", ex.get_tile_data(tiles[0]).shape) # expect (576, 576, 13)
PY
```

---

## 3. What NOT to do

**Never put the repo under `~/Documents`.** It's iCloud-managed: the file provider returns
`EPERM` for venv creation and some `mkdir` calls, and it evicts cold files to `.icloud`
placeholders that `rasterio` cannot open. This cost a full day. Keep it at `~/RA_Raji/`.

**Never move or rename the repo without rebuilding anything inside it.** Venvs hardcode
absolute paths into `bin/activate` and all 58 console scripts. After a move the prompt
still shows `(venv)` while `python` silently resolves elsewhere and every import fails.

**Never trust `(venv)` in the prompt.** `~/.rye/shims` sits ahead of the venv on `PATH`.
Diagnose with `which python && python -VV`, always.

**Never run `uv init`, `uv sync` or `uv add` here.** This is a `requirements.txt` project.
`uv init` writes a `pyproject.toml` pinned to `>=3.13`, a `.python-version` of `3.13`, and
a second venv at `.venv/` — after which nothing resolves. Only
`uv pip install -r requirements-lock_2026-07-17.txt` is safe.

**Never use `caffeinate`, and never run all 16 periods in one process.** That combination —
full CPU, no rest, machine forced awake for ten hours — is what overheated the laptop.
The pipeline is resume-safe, so sleeping mid-run costs nothing.

**Never run `python timeseries/run_trials.py` directly without `export EE_PROJECT`.**
`.env` is invisible to Python (`gee.py:20` falls back to upstream's `earthindex`), and the
resulting permission error never names your project. `run_staged.sh` handles this for you.

**Never change `clear_threshold` without a fresh `--cache_dir`.** It's baked into the
composite *and* absent from the cache key, so stale tiles are reused silently and the
results are contaminated with no error. The mining threshold and the observed/missing
threshold, by contrast, are free to change afterwards — `prob` and `clear_frac` are stored
continuously for every chip.

**Never mix panels from different configurations in one `--panels_dir`.** `analysis.py`
globs everything with no filter. It now refuses on mixed `clear_threshold` and warns on
mixed model or schema, but keep separate runs in separate `--out_root`s anyway.

---

## 4. If something breaks

| symptom | cause → fix |
|---|---|
| `ModuleNotFoundError` while `(venv)` shows | wrong interpreter — `which python`, re-activate |
| `Operation not permitted` on mkdir/venv | path is under `~/Documents` (iCloud) |
| resolver demands `numpy>=2.3.3` | you targeted Python >3.11 |
| browser says *Access blocked* during auth | use `auth_mode='localhost'` (§2 step 3) |
| GEE error naming a project you don't recognise | `EE_PROJECT` unset → defaulted to `earthindex` |
| many `SSLEOFError` retries, batches 40× slower | network degradation, not CPU. No timeout exists — Ctrl-C and re-run |
| `rasterio` hangs opening a cached tile | cache is in a synced folder; move it |

Full symptom table with root-cause evidence: `CODE_WALKTHROUGH.md` §0.9.
