<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# Run 2 (2-year pilot) — setup, resource audit, threshold provenance, pipeline internals

Written 2026-08-04. Environment section closed out 2026-08-04 evening.

**Every claim below is sourced to a file:line or to a log/artifact on disk.** Where a
fact could not be verified it is marked `NOT DETERMINABLE`. Where documentation
contradicts the code, the code wins and the contradiction is listed in §6.

Verification base: working tree at `main` = `02683f9`, upstream/main = `bbbcb2d`.

**Companion document:** `docs/CODE_WALKTHROUGH.md` Part 0 was rewritten from this incident
log and is the operational setup manual. This document is the *evidence*; Part 0 is the
*procedure*. Keep them in sync.

---

## 1. Environment — the full incident log

**Six independent faults stacked on top of each other.** Each masked the next, which is
why the errors kept changing shape across the day. Listed in the order diagnosed, with the
evidence for each.

| # | Fault | Symptom seen | Status |
|---|---|---|---|
| A | venv built at the old repo path | `No module named 'ee'` / `'geopandas'` with `(venv)` showing | **solved** — rebuild |
| B | global `~/.venv` on Python 3.14 active | `Using Python 3.14.0 environment at: /Users/…/.venv`; `pandas`/`numpy` resolution failure | **solved** — `unset VIRTUAL_ENV`; shell config still to check |
| C | `uv init` scaffolded a conflicting project | `incompatible with the project's Python requirement: >=3.13` | **solved** — artifacts deleted |
| D | iCloud folder cannot host a venv | `Errno 1 Operation not permitted`; `failed to make path absolute` | **eliminated** — repo relocated out of iCloud 2026-08-07 |
| E | `rye` shims ahead of the venv on `PATH` | `which python` → `~/.rye/shims/python` | **open** — verify with `which python`, or uninstall rye |
| F | `ee.Authenticate()` blocked by Google | browser returns *Access blocked*; drive-scope warning | **solved** — `auth_mode='localhost'` + reduced scopes |

**The single lesson, if you read nothing else:** `(venv)` in the prompt is not evidence.
Faults A, B and E all present identically — packages that are demonstrably installed report
as missing — and all three are diagnosed in one command: `which python && python -VV`.

### 1.1 Fault A — the venv was built at the old repo path (SOLVED)

At the time of this fault the repo had been moved from `~/RA_Raji/AMW-mining-detector`
into `~/Documents/RA_Raji/AMW-mining-detector` (it was moved back on 2026-08-07 — §1.4).
The venv was created before that move and hardcoded the old absolute path:

```
venv/bin/activate:81   VIRTUAL_ENV='/Users/rubygrambauer/RA_Raji/AMW-mining-detector/venv'
venv/bin/earthengine   #!/Users/rubygrambauer/RA_Raji/AMW-mining-detector/venv/bin/python
venv/bin/pip           #!/Users/rubygrambauer/RA_Raji/AMW-mining-detector/venv/bin/python
```

`source venv/bin/activate` therefore prepended a **non-existent** directory to `PATH`,
printed `(venv)` in the prompt, and left `python` resolving to whatever was next on
`PATH`. All 58 console scripts in `venv/bin/` had the same dead shebang.

**This is the original cause of `ModuleNotFoundError: No module named 'ee'`** — the
packages were present in `venv/lib/python3.11/site-packages/` the whole time; the
interpreter running was not that venv's.

**Symptoms it produced:** `No module named 'ee'`, `No module named 'geopandas'`,
`zsh: command not found: earthengine`.

**Lesson:** uv/virtualenv venvs are **not relocatable**. Moving or renaming a repo
invalidates its venv. Rebuild, never move.

### 1.2 Fault B — a global `~/.venv` on Python 3.14 was active (OPEN)

```
$ uv pip install -r requirements-lock_2026-07-17.txt
Using Python 3.14.0 environment at: /Users/rubygrambauer/.venv
```

A venv at `~/.venv` (Python 3.14.0) was active and shadowing everything. This is what
`(venv)` in the prompt actually referred to — including back in `core/` when the first
`import ee` failed.

**Symptom it produced:** the dependency resolution failure

```
Because pandas==3.0.3 depends on numpy>=2.3.3 and you require numpy==1.26.4,
we can conclude that your requirements and pandas==3.0.3 are incompatible.
```

This is not a bad lock file. uv resolved against **Python 3.14**, where `pandas 3.0.3`
declares a numpy floor of `>=2.3.3`. On Python 3.11 the floor is lower — the lock file is
itself the evidence, having been frozen from a working 3.11 environment on 2026-07-17.

**Decisive constraint regardless of numpy: `tensorflow==2.14.0` publishes no wheels above
cp311.** Python 3.13 and 3.14 are both impossible for this project. **Python 3.11 is
mandatory.**

**Still to do:** find out why `~/.venv` auto-activates.
`grep -n "venv\|VIRTUAL_ENV" ~/.zshrc ~/.zprofile ~/.zshenv`. Until that line is removed,
every new terminal re-shadows the repo venv.

### 1.3 Fault C — `uv init` scaffolded a conflicting project (SOLVED)

`uv sync` and `uv add` were the wrong commands: this repo has **no `pyproject.toml`** and
manages dependencies via `requirements.txt` (loose) + `requirements-lock_2026-07-17.txt`
(pinned). `uv sync` requires a `pyproject.toml`; `uv add` failed only because no package
argument was given.

Running `uv init` to create one made things worse. At 15:52 it wrote:

| file | content | effect |
|---|---|---|
| `pyproject.toml` | `requires-python = ">=3.13"`, `dependencies = []` | forces 3.13; a future `uv sync` would install nothing and could uninstall what is there |
| `.python-version` | `3.13` | makes bare `uv venv` default to 3.13 |
| `uv.lock` | empty virtual project | — |
| `main.py` | `print("Hello from amw-mining-detector!")` | junk at repo root |
| `.venv/` | Python **3.13.5**, `prompt = amw-mining-detector` | a *second* venv in the repo, alongside `venv/` |

**Symptom it produced:**

```
warning: The requested interpreter resolved to Python 3.11.15, which is incompatible
with the project's Python requirement: `>=3.13` (from `project.requires-python`)
```

All five artifacts were untracked (`git status`: `??`) and must be deleted:

```bash
rm -f pyproject.toml .python-version uv.lock main.py
rm -rf .venv
```

Note: a *correct* `pyproject.toml` (`requires-python = ">=3.11,<3.12"` + the full pinned
dependency set) would be a real reproducibility improvement for the paper. That is a
deliberate migration to do **after** the pilot runs — not the empty `uv init` stub.

### 1.4 Fault D — the repo directory cannot host a venv (SOLVED — root cause)

```
$ uv pip install --python venv/bin/python -r requirements-lock_2026-07-17.txt
error: Failed to inspect Python interpreter from provided path at `venv/bin/python`
  Caused by: Querying Python at `venv/bin/python` failed with exit status 1
[stderr]
Exception ignored error evaluating path:
  File "<frozen getpath>", line 268, in <module>
OSError: failed to make path absolute
Fatal Python error: error evaluating path
Python runtime state: core initialized
```

**What is verifiable:** the venv itself is structurally correct.

```
venv/pyvenv.cfg          version_info = 3.11.15, uv = 0.11.21
venv/bin/activate:81     VIRTUAL_ENV='/Users/rubygrambauer/RA_Raji/AMW-mining-detector/venv'  ← correct path now
venv/bin/python       -> /Users/rubygrambauer/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11
venv/lib/python3.11/site-packages/   _virtualenv.pth only  ← EMPTY, install never ran
```

The interpreter **starts** ("Python runtime state: core initialized") and then fails while
computing its own paths, so the symlink target exists and the binary is executable.
`OSError: failed to make path absolute` originates in CPython's path bootstrap, which
makes relative paths absolute using `getcwd()`.

**Root cause, established by three tests run 2026-08-04:**

```bash
$ "$UVPY" -c "import os; print(os.getcwd())"
getcwd OK: /Users/rubygrambauer/RA_Raji/AMW-mining-detector   # ← getcwd is FINE

$ "$UVPY" -m venv /tmp/venvtest
TMP VENV OK                                                            # ← interpreter is FINE

$ "$UVPY" -m venv venv          # inside the repo
Error: [Errno 1] Operation not permitted                               # ← the DIRECTORY refuses
```

`~/Documents` is **iCloud-managed** (Desktop & Documents Folders sync). The iCloud file
provider does not support the operations `venv` creation requires — symlinking and setting
file flags on the copied interpreter. `EPERM` is returned by the file provider, not by
Unix permissions, so `chmod`/`sudo` cannot fix it.

The earlier `OSError: failed to make path absolute` was the same wall from the other side:
uv had created a venv whose `bin/python` symlink the file provider would not resolve for a
spawned process, so CPython's path bootstrap died.

**The decisive asymmetry: `getcwd()` works.** Code can be *executed* from inside the iCloud
folder without problems — only the venv cannot *live* there.

Historical confirmation: the original venv worked when the repo was at
`~/RA_Raji/AMW-mining-detector`, outside `~/Documents`. Every venv fault in this log dates
from the move into iCloud.

> ### ✅ ELIMINATED 2026-08-07 — repo relocated out of iCloud
>
> The repository was moved back to **`~/RA_Raji/AMW-mining-detector`**. Verified after the
> move: `git fsck` clean, `git log` intact at `c3f432a`, 7.9 GB present, and
> `run_staged.sh --dry-run` created `data/timeseries_run2/` **successfully** — the `mkdir`
> that returned `EPERM` under iCloud now works.
>
> The move was low-risk *specifically because* the venv, tile cache and run outputs had
> already been relocated outside the repo. Nothing inside the folder hardcoded an absolute
> path; both shell scripts derive the repo root from `$0`.
>
> **What this changes:**
> - Run outputs may now live inside the repo again (`data/timeseries_run2/`) — the default.
> - The tile cache *may* move back, but §1.6c's advice still stands for a different reason:
>   5,120 files of 4.8 GB/year is noise in `git status` and bulk in any backup.
> - The venv at `~/venvs/amw` **can** move back into the repo, but there is no reason to.
>   Keeping it out also immunises the setup against a future repo move (§1.1).
>
> Rows in `CODE_WALKTHROUGH.md` §0.9 tagged `[2026-08-04]` for iCloud `EPERM` are now
> historical. They are kept because the failure mode is non-obvious and recurs the moment
> anything lands in `~/Documents` again.

### 1.5 Fault E — `rye` shims sit ahead of everything on PATH (OPEN)

Revealed while diagnosing Fault A:

```
$ which python
/Users/rubygrambauer/.rye/shims/python
```

`rye` is installed and `~/.rye/shims` is on `PATH` **ahead of** the venv. This, not just
`~/.venv`, is why bare `python` kept resolving to the wrong interpreter — the rye shim
intercepts `python` and resolves it to whatever rye considers the active environment
(here, `~/.venv` on 3.14).

A venv with a **valid** path still wins, because `activate` prepends its `bin/` ahead of
the shims — confirmed: after `source ~/venvs/amw/bin/activate`, `which python` correctly
gives `/Users/rubygrambauer/venvs/amw/bin/python`. But a venv with a **dead** path
(Fault A) silently fell through to the shim. That combination is what made the original
error so opaque.

**Still to do:** decide whether rye is wanted at all. If not,
`rye self uninstall` and remove the `~/.rye/shims` line from `~/.zshrc` / `~/.zprofile`.
If it is wanted, always verify with `which python` — never trust `(venv)` in the prompt.

### 1.6 The working setup — venv outside iCloud

**Do you need uv? No.** uv is only a package installer here. This route uses it once to
locate a 3.11 interpreter, then drops it.

```bash
# clean up the failed attempts (untracked scaffolding + broken venvs)
cd ~/RA_Raji/AMW-mining-detector
deactivate 2>/dev/null; unset VIRTUAL_ENV
rm -f pyproject.toml .python-version uv.lock main.py
rm -rf .venv venv

# create the venv OUTSIDE iCloud
UVPY=$(uv python find 3.11)
"$UVPY" -VV                                   # must print 3.11.x
mkdir -p ~/venvs
"$UVPY" -m venv ~/venvs/amw

source ~/venvs/amw/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/RA_Raji/AMW-mining-detector/requirements-lock_2026-07-17.txt
```

`~/venvs` is a plain local path: no file provider, no TCC scope, no sync. Verified working
2026-08-04 (`python -VV` → 3.11.15, `which python` → `/Users/rubygrambauer/venvs/amw/bin/python`).

### 1.6b Consequences of the venv living outside the repo

Two places assume `venv/` sits in the repo root and will now fail:

1. **`timeseries/run_overnight.sh:19`** — `source venv/bin/activate`. Change to
   `source ~/venvs/amw/bin/activate`, or export a variable and use it.
2. **`README.md:51`** — "`source venv/bin/activate`". Stale; update to the new path.

Activate manually from now on:

```bash
source ~/venvs/amw/bin/activate
cd ~/RA_Raji/AMW-mining-detector
```

### 1.6c iCloud and the tile cache — a real hazard for the long run

The cache is **4.8 GB per year** (§2.5, measured). Writing that into an iCloud-synced
folder means continuous upload churn during the run, and — worse — iCloud may **evict**
cold files to placeholders (`*.icloud`), at which point `rasterio.open()` on a cached tile
fails or blocks on a network fetch. `data/timeseries/cache/` is gitignored but **not**
excluded from iCloud.

`run_trials.py:76` exposes `--cache_dir`. Point it outside iCloud:

```bash
mkdir -p ~/amw_cache
python timeseries/run_trials.py \
  --out_root data/timeseries_run2 \
  --cache_dir ~/amw_cache/2025 \
  --year 2025 …
```

Panels and analysis outputs (~27 MB/year) can stay in the repo — small, and you want them
synced and backed up.

### 1.7 Verification gate — do not skip

```bash
python -VV                         # must be 3.11.x
which python                       # must be /Users/rubygrambauer/venvs/amw/bin/python
python -c "import ee, geopandas, tensorflow, numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

Expected: `3.11.15`, the `~/venvs/amw` path, and `1.26.4 3.0.3`. If `which python` shows
`~/.rye/shims/python`, Fault E (§1.5) has re-asserted itself — re-activate before doing
anything else.

### 1.8 `.env` is not read by any Python code — silent wrong-project failure

`.env` line 1 is:

```
EE_PROJECT = "deforestation-and-mining"
```

`core/gee.py:20` reads:

```python
EE_PROJECT = os.environ.get("EE_PROJECT", "earthindex")
```

**No Python module in this repo calls `load_dotenv`; `python-dotenv` is not installed.**
Only `timeseries/run_overnight.sh:22-27` parses `.env`, with a `sed` workaround because
the file is not valid shell (spaces around `=`, quotes).

Consequence: running `python timeseries/run_trials.py` without exporting `EE_PROJECT`
initialises Earth Engine against **`earthindex`** — upstream's project, not yours. The
resulting permission error never mentions your project. This is the most likely source of
confusing GEE errors after a project switch.

The previous run used project `arboreal-pen-334513` (visible in the retry URLs in
`data/timeseries/runlog_20260728_223615.log`). The switch to `deforestation-and-mining`
is real and is only picked up via `.env` → `run_overnight.sh`, or via explicit `export`.

### 1.9 Auth mechanism (verified against the installed library)

`gee.py:60` calls `ee.Initialize(opt_url=..., project=EE_PROJECT)` with no credentials,
unless `GOOGLE_APPLICATION_CREDENTIALS` points at a service-account JSON
(`gee.py:43-59`). With no credentials argument, `ee/__init__.py:230` →
`ee/data.py:203 get_persistent_credentials()`, which (verified at `ee/data.py:214-231`):

1. reads `~/.config/earthengine/credentials` (written by `earthengine authenticate`);
2. falls back to `google.auth.default()` (ADC, from `gcloud auth application-default login`).

Both paths in `README.md:53-56` are valid. You need **one**.

`earthengine` is a console script shipped by `earthengine-api`, so it only exists once the
venv is genuinely active — `zsh: command not found: earthengine` is a symptom of Fault A,
not a missing package. `python -c "import ee; ee.Authenticate()"` is the equivalent.

#### Fault F — bare `ee.Authenticate()` is blocked by Google (SOLVED)

```
Fetching credentials using gcloud
WARNING: The following scopes will be blocked soon for the default client ID:
https://www.googleapis.com/auth/drive
```
→ browser returns **Access blocked**.

Causal chain, verified in `earthengine-api 1.7.35` source:

1. `oauth.py:550-558` auto-selects the auth mode. `_no_gcloud()` (`oauth.py:397`) returns
   False because `gcloud` is on `PATH`, so it falls through to `auth_mode = 'gcloud'`.
2. `gcloud` mode (`oauth.py:401 _load_gcloud_credentials`) shells out to gcloud, which uses
   **gcloud's own OAuth client ID** `764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur`
   (confirmed in the URL printed to the terminal).
3. Google blocks the `drive` scope for that client ID → the flow is refused.

`earthengine-api` ships its **own** client ID
`517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359` (`oauth.py:43-45`), used by the
`localhost` and `notebook` modes. Forcing `localhost` bypasses gcloud entirely.

**Fix:**

```bash
python - <<'PY'
import ee
ee.Authenticate(
    auth_mode='localhost',
    scopes=['https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform'],
    force=True)
PY
```

`localhost` mode defaults to port 8085 (`oauth.py:607`); use `auth_mode='localhost:0'` to
auto-select a free port if 8085 is occupied.

**Dropping the `drive` and `devstorage` scopes is safe — verified, not assumed:**

- `core/gee.py:55` requests exactly `["https://www.googleapis.com/auth/earthengine"]` on
  the service-account path. All data flows through `ee.data.computePixels`. Nothing in
  this pipeline touches Drive or Cloud Storage.
- Reduced scopes do **not** break `ee.Initialize()`. `data.py:237` calls
  `oauth.is_valid_credentials(credentials)`, whose signature (`oauth.py:154`) takes **no**
  `scopes` argument and only attempts a token refresh. The scope check lives solely in
  `_valid_credentials_exist()` (`oauth.py:139-150`), which is used only by `authenticate()`
  to decide whether to skip the flow.

Fallback if `localhost` mode cannot reach a browser on the same host:
`auth_mode='notebook'`. Note `oauth.py:510` — **notebook credentials expire in 7 days**,
so it is unsuitable for an unattended overnight run.

### 1.10 Full working sequence, once §1.7 passes

```bash
source ~/venvs/amw/bin/activate                 # venv lives OUTSIDE iCloud (§1.4)
cd ~/RA_Raji/AMW-mining-detector
python -c "import ee; ee.Authenticate()"        # once per machine
export EE_PROJECT=deforestation-and-mining      # .env is NOT read by Python

# end-to-end auth + data check (one 552px tile, one month)
python - <<'PY'
import sys
sys.path.insert(0, "core")
import gee, tile_utils, geopandas as gpd
cfg = gee.DataConfig()
ex  = gee.GEE_Data_Extractor("2025-07-01", "2025-08-01", cfg)
region = gpd.read_file("data/boundaries/la_pampa_corridor.geojson").union_all()
tiles  = tile_utils.create_tiles(region, cfg.tilesize, cfg.pad)
print("tiles in AOI:", len(tiles))
px = ex.get_tile_data(tiles[0])
print("tile shape:", px.shape, "dtype:", px.dtype, "nonzero frac:", float((px != 0).mean()))
PY
```

Expected: `tiles in AOI: 80`, `tile shape: (576, 576, 13)`.
`576 = 552 + 2×12` (`gee.py:68-69`, `gee.py:190`).

### 1.11 Standing rules extracted from this incident

0. **The repo lives at `~/RA_Raji/AMW-mining-detector`. Keep it out of `~/Documents`.**
   That path is iCloud-managed; its file provider returns `EPERM` for venv creation and
   for some `mkdir` calls, and it evicts cold files to `.icloud` placeholders that
   `rasterio` cannot open.
1. **The venv lives at `~/venvs/amw`, never inside the repo.** Now a choice rather than a
   constraint, and still the right one: it survives any future repo move.
2. **Never move or rename the repo directory** without rebuilding anything inside it.
   Venvs hardcode absolute paths in `bin/activate` and in every console-script shebang;
   moving the repo silently breaks them while still printing `(venv)`. The 2026-08-07
   relocation was safe *only* because nothing path-dependent remained inside the folder.
3. **`(venv)` in the prompt proves nothing.** `~/.rye/shims` sits ahead of the venv on
   `PATH` (§1.5). Always verify with `which python && python -VV`.
4. **Do not run `uv init` / `uv sync` / `uv add` in this repo.** It is a
   `requirements.txt` project, not a uv project. `uv init` creates a `pyproject.toml`
   pinned to `>=3.13` plus a second venv at `.venv/`, and breaks everything.
5. **Python 3.11 is mandatory** (`tensorflow==2.14.0` publishes no wheels above cp311).
   Any resolution error mentioning `numpy>=2.3.3` means the wrong interpreter was targeted.
6. **Always `export EE_PROJECT`.** `.env` is invisible to Python (§1.8).
7. **Keep the tile cache out of iCloud** — `--cache_dir ~/amw_cache/<year>` (§1.6c).
   4.8 GB/year of `.tif`s in a synced folder risks eviction mid-run.
8. **Add to `.gitignore`:** `.venv/`, `pyproject.toml`, `uv.lock`, `.python-version`,
   `main.py` — or at minimum do not commit them.

---

## 2. The tiny 2-year test — what the CLI actually supports

### 2.1 `run_trials.py` takes ONE year

`timeseries/run_trials.py:74`: `--year`, `type=int`, `default=2025`. There is no
multi-year flag. A 2-year run is **two invocations**. The cache directory defaults to
`<out_root>/cache/<year>` (`run_trials.py:92`), so the two years do not collide.

### 2.2 Verified flags (README claims match code here)

| flag | default | line |
|---|---|---|
| `--region` | `data/boundaries/la_pampa_corridor.geojson` | `run_trials.py:70` |
| `--model` | `models/48px_v4.10b-…-ensemble.h5` | `:72` |
| `--year` | `2025` | `:74` |
| `--out_root` | `data/timeseries` | `:75` |
| `--cache_dir` | `<out_root>/cache/<year>` | `:76`, `:92` |
| `--threshold` | `0.4` — **only** gates the convenience GeoJSON | `:78`, `panel.py:245` |
| `--max_workers` | `8` | `:81` |
| `--limit_tiles` | `None` | `:82` |
| `--months` | `None` (comma list `2025-07,2025-08`) | `:84` |
| `--no-monthly` / `--no-quarterly` | off | `:86-87` |
| `--force` | off | `:88` |

There is **no** `--limit_area`, no AOI-subsetting flag, and no `--quarters` flag. To
shrink the area you either pass a smaller `--region` GeoJSON or use `--limit_tiles N`
(which takes the *first* N tiles in DLTile iteration order — a contiguous-ish strip, not
a random sample; `panel.py:130`).

### 2.3 Blocking issue for a 2-year run: `analysis.py` pools everything

`timeseries/analysis.py:94`: `files = sorted(panels_dir.glob("panel_*.parquet"))`.
It globs **every** panel in the directory with no year filter.
`data/timeseries/panels/` currently holds all 16 panels from the completed 2025 run
(28 MB, mtimes 2026-07-28/29).

If you write run-2 panels into the same directory, `analysis.py` will silently pool
2025 with the new years. **Use a separate `--out_root` for run 2.**

### 2.4 Recommended smoke test (≈2–3 min, 2 tiles, 1 month)

```bash
export EE_PROJECT=deforestation-and-mining
python timeseries/run_trials.py \
  --out_root data/timeseries_run2 \
  --year 2025 --months 2025-07 --no-quarterly \
  --limit_tiles 2 --max_workers 2
```

Success criteria (all checkable on disk):

- `data/timeseries_run2/panels/panel_m2025-07.parquet` exists,
  `2 tiles × 529 chips = 1058` rows.
- `data/timeseries_run2/tiles/tile_stats_m2025-07.csv` has `status,OK` for both tiles and
  `fill0_mismatch` ≈ `0.0` (the empirical check of the "EE fills gaps with 0" assumption —
  `panel.py:75-88`).
- `data/timeseries_run2/cache/2025/` holds 4 files: 2 × `S2L1C_*.tif` + 2 × `S2COV_*.tif`.

Then delete the 2-tile stub before the real run:
`rm data/timeseries_run2/panels/panel_m2025-07.parquet` (the cached `.tif`s can stay —
they are reused).

> **STATUS 2026-08-04: smoke test NOT YET CONFIRMED.** As of the last inspection,
> `data/timeseries_run2/` does not exist, no new `panel_*.parquet` has been written, and
> no new `runlog_*.log` is present. The environment (§1) is fixed and Earth Engine
> authentication completed, but there is no artifact on disk proving an end-to-end run.
> **Do not treat the pipeline as verified until the four success criteria above are
> checked.** Record the result here when it passes.
>
> Two anomalies to resolve while doing so:
>
> 1. `core/__pycache__/gee.cpython-312.pyc` and `timeseries/__pycache__/panel.cpython-312.pyc`
>    exist with recent mtimes. **`312` = Python 3.12**, not the 3.11 venv at `~/venvs/amw`.
>    Something imported `gee.py` and `panel.py` under a 3.12 interpreter. `tensorflow==2.14.0`
>    cannot run there (no wheels above cp311), so any run through that interpreter cannot have
>    completed inference. Clear them and re-check:
>    `find . -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +`, then confirm the
>    new `.pyc` files are `cpython-311`.
> 2. Stale `.pyc` files from earlier interpreters (`cpython-310`) are still scattered through
>    `scripts/`, `utils/` and `timeseries/`. Harmless, but they make interpreter forensics
>    harder. Same command clears them.

### 2.5 Scale of the real 2-year run (extrapolated from measured run 1)

Measured on `la_pampa_corridor` (1,900 km², **80 tiles**, 529 chips/tile,
42,320 chips/period):

| | per year |
|---|---|
| periods | 16 (12 monthly + 4 quarterly) |
| cache files | 2,560 (= 16 × 80 × 2) |
| cache size | **4.8 GB** (measured, `data/timeseries/cache/2025`) |
| panels | 16 × ~1.7 MB = 27 MB |

**For 2 years at full AOI: ~9.6 GB cache, ~5,120 files.** Halving the AOI halves both.

> **Cache duplication, found 2026-08-04.** Run 1's cache still sits at
> `data/timeseries/cache/2025/` (4.8 GB, 2,560 files). The cache key is
> `{collection}_{tile.key}_{start}_{end}.tif` (`gee.py:178`) — it encodes **neither the
> AOI nor `clear_threshold`**. Run 2 uses identical tiling defaults, the same AOI and the
> same date windows, so **the filenames are identical** and run 1's tiles are directly
> reusable. Merge before any re-run:
>
> ```bash
> rsync -a --ignore-existing \
>   <repo>/data/timeseries/cache/2025/ ~/amw_cache/2025/
> ```
>
> `--ignore-existing` never overwrites; rsync writes to a temp name and renames, so this
> is safe even while a job is running. Without it, the schema-v2 `--force` re-run would
> re-fetch every tile from Earth Engine for no benefit.

### 2.6 Panel schema v2 — implemented 2026-08-04

`PANEL_SCHEMA_VERSION = 2` (`panel.py`). Two additions, both irrecoverable if not
captured at inference time.

**Per-member probabilities `p_m0 … p_m5`.** Verified by reading the model_config of
`models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`: the model is a Keras `Concatenate` of
six `Sequential` sub-models, each ending in one sigmoid unit —

```
48px_v4.10-CNN800kb  48px_v4.18-CNN800kd  48px_v4.20-CNN800kg
48px_v4.21-CNN100ka  48px_v4.22-CNN800kb  48px_v4.22-CNN800kc
```

so `_model_infer` returns `(n, 6)` and `_probs_from_predictions`
(`core/inference_engine.py:632`) collapses it with `np.mean(axis=1)`. Panel v1 kept only
the mean. The member spread is the only uncertainty signal this model emits and is
directly relevant to the research question — *do the six models disagree more in cloudy
months?* is a sharper instrument-stability metric than the mean alone.

We store the **raw six**, not std/min/max, for the same reason the panel stores continuous
`prob` rather than a thresholded decision: summarising is a modelling choice, better made
in `analysis.py` where it can be revised.

Measured cost (from `panel_m2025-01.parquet` parquet metadata, not estimated):

| | v1 | v2 |
|---|---|---|
| `prob` column, 42,320 rows, compressed | 339 KB | 339 KB |
| six member columns | — | ~2.0 MB |
| **panel file** | **1.68 MB** | **~3.7 MB** |
| **two years (32 periods)** | **54 MB** | **~118 MB** |

Extra compute: **zero** — the array already exists in memory. Extra RAM in `analysis.py`:
~33 MB per year, against ~130 MB already used by geometry. Against a 4.8 GB/year tile
cache the storage delta is 1.3 %. `prob` remains bit-exactly `mean(p_m*)` — verified.

**Provenance columns:** `schema_version, model, model_sha256, ensemble_members,
n_members, clear_threshold, collection, tilesize, pad, chip_size, stride, git_commit`.
Constant per file, so parquet dictionary-encodes each to a few hundred bytes (`period` and
`cadence` measure 0.2 KB). Necessary because `analysis.py:94` globs **every**
`panel_*.parquet` in a directory with no filter — panels from different models or cloud
thresholds would otherwise pool silently. `clear_threshold` matters most: it is baked into
the composite *and* absent from the tile-cache key, so it cannot be inferred afterwards.

`git_commit` carries a `-dirty` suffix when the tree has uncommitted changes (currently
`02683f9-dirty`). **Commit before the production run** so panels cite a clean hash.

**Which thresholds are still free to change after a run:**

| threshold | changeable post hoc? | why |
|---|---|---|
| `pred_threshold` / `T_MAIN` (0.4) | **yes, freely** | `prob` stored continuous for every chip (`panel.py:206-216`); the flag only gates the convenience GeoJSON (`panel.py:245`) |
| `TAU_OBSERVED` (0.95) | **yes, freely** | `clear_frac` stored per chip |
| ensemble aggregation (mean / vote / trimmed) | **yes, from v2 on** | raw members stored |
| `clear_threshold` (0.6) | **no** | `gee.py:132` masks scenes *before* the median, so pixel values change; requires a full re-fetch into a separate `--cache_dir` |

---

## 3. Laptop protection — audit result: **nothing is in place**

### 3.1 What I searched for and did NOT find

Grepped `core/`, `timeseries/`, `scripts/`, `utils/` for:
`OMP_NUM_THREADS`, `inter_op_parallelism`, `intra_op_parallelism`,
`set_memory_growth`, `LogicalDeviceConfiguration`, `resource.setrlimit`,
`torch.set_num_threads`, `nice`, `cpulimit`, `VECLIB_MAXIMUM_THREADS`,
`OPENBLAS_NUM_THREADS`, any thermal or power reference.

**Zero matches outside `venv/` and `_archive/`.** There is no CPU thread cap, no memory
cap, no process-priority setting, no thermal monitoring, and no throttling anywhere in
this codebase.

### 3.2 What exists, and why it makes things worse rather than better

`timeseries/run_overnight.sh:44-46`:

```bash
if command -v caffeinate >/dev/null; then
  nohup caffeinate -im -w "$JOB_PID" >/dev/null 2>&1 &
fi
```

`caffeinate -i` prevents idle sleep, `-m` prevents disk sleep. This is a **sustained-load
enabler**, not a safeguard: it guarantees the machine cannot fall back to a low-power
state for the whole run. Combined with `run_overnight.sh:49` ("keep the laptop plugged
in, lid open"), the documented operating mode is: full CPU load, charging, no sleep, for
hours. That is the configuration under which your machine overheated.

### 3.3 Where the heat actually comes from

Two independent load sources, both uncapped:

1. **Network/JSON decode**, `panel.py:165-171` — a `ThreadPoolExecutor(max_workers=8)`
   submitting **16 concurrent EE requests** per batch (8 imagery + 8 coverage;
   `panel.py:166` and `:168` both submit for every tile in the batch). Each response is
   a 576×576×13 float32 NumPy array ≈ 17 MB, plus 576×576×2 uint16 for coverage.
   Peak resident: ~8 × 17 MB ≈ 140 MB for imagery. Not memory-critical; it is the
   decompression + array assembly that burns CPU.

2. **TensorFlow inference**, `panel.py:192` → `inference_engine.py:616-620`
   `self.model.predict(x, verbose=0)` under `self._tf_model_lock`. Serialised across
   tiles, but **TF itself uses all available cores** — no `intra_op` limit is set. This
   is the dominant sustained load: measured `infer_s ≈ 7.5 s` per tile
   (`data/timeseries/tiles/tile_stats_m2025-01.csv`) vs `fetch_s ≈ 2.4 s` amortised.

   Aggravating factor: `core/tf_darwin.py:26` calls
   `tf.config.run_functions_eagerly(True)` at import (`inference_engine.py:40`,
   `model_library.py:10`). This **disables graph-mode optimisation** — it exists to avoid
   a Metal/tf.data kernel crash on macOS, at the cost of slower, less efficient
   inference. So the CPU does more work per chip than it needs to.

### 3.4 Concrete mitigations (none currently implemented)

| # | Measure | How | Effect |
|---|---|---|---|
| 1 | Cap TF threads | `export TF_NUM_INTRAOP_THREADS=4 OMP_NUM_THREADS=4`, or `tf.config.threading.set_intra_op_parallelism_threads(4)` | Directly caps the dominant heat source. On an 8-core machine roughly halves package power at ~1.6× wall-clock. |
| 2 | Lower `--max_workers` | `--max_workers 4` | Halves concurrent EE requests *and* per-batch tile count (`panel.py:161` reuses `max_workers` as the batch size — one knob, two effects). |
| 3 | Deprioritise the process | `nice -n 10 python timeseries/run_trials.py …` | Lets macOS schedule around it; on Apple Silicon, low-priority work is preferentially placed on efficiency cores. Large thermal effect, no code change. |
| 4 | Drop `caffeinate -m` | `run_overnight.sh:45` → `caffeinate -i -w` | Allows disk sleep. Minor. |
| 5 | Inter-batch pause | `time.sleep(N)` after `panel.py:228` | Duty-cycles the CPU. Crude but effective; costs wall-clock proportionally. |
| 6 | Thermal watchdog | external loop on `sudo powermetrics --samplers smc`; `kill -STOP`/`-CONT` around a limit | The only actual closed-loop protection. Requires sudo. Not written. |

### 3.5 IMPLEMENTED 2026-08-04 — what changed in the code

Measures 1, 2, 3 and 5 are now implemented. Two new fork-original files, one two-line edit,
one bug fix. **No file under `core/` was touched**, so `git checkout upstream/main -- core/`
remains safe (GIT_PLAYBOOK.md).

| file | change |
|---|---|
| `timeseries/resource_limits.py` | **NEW.** Caps CPU parallelism. Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` (via `setdefault`, so a shell value always wins), then `tf.config.threading.set_intra_op_parallelism_threads(n)` / `set_inter_op_parallelism_threads(n//2)`, then `torch.set_num_threads(n)`. Budget from `AMW_THREADS`, default 4; `0` restores stock behaviour. |
| `timeseries/run_trials.py:43-47` | **EDIT.** Calls `apply_resource_limits()` **before** `from panel import …`. Ordering is load-bearing: the BLAS/OpenMP variables are read at library import time, and `tf.config.threading.set_*` raises `RuntimeError` once the TF context exists. `panel → inference_engine → tensorflow`, so this is the last safe point. |
| `timeseries/run_trials.py:119` | **EDIT.** Logs the effective caps into the runlog, so a slow run can be *attributed* rather than guessed at. |
| `timeseries/run_staged.sh` | **NEW.** Runs one period per `run_trials.py` invocation with an idle cooldown between, under `taskpolicy -b` (macOS background QoS) or `nice`. Verifies the interpreter is 3.11 before starting, resolves `EE_PROJECT` from `.env`, and fails with an actionable message if the output directory cannot be created (iCloud `EPERM`, §1.4). |

**Cooldown design — sensor-free by default.** `pmset -g therm` exposes `CPU_Speed_Limit` on
Intel Macs but reported nothing parseable on this Apple Silicon machine (observed
2026-08-04: `thermal: unknown`), and there is no no-sudo temperature API there. The primary
mechanism is therefore **duration-proportional**: rest for `AMW_COOLDOWN_PCT` (default 75 %)
of the time the period just took, floored at `AMW_COOLDOWN` (default 600 s). Measured
periods map to:

| period took | rest |
|---|---|
| 2 min (smoke test) | 10 min (floor) |
| 12 min (typical month) | 10 min (floor) |
| 28 min (typical quarter) | 21 min |
| 40 min (degraded) | 30 min |

Duty cycle ≈ 57 % at the 75 % ratio. The thermal probe, where it works, can only **extend**
this cooldown, never shorten it — so a missing sensor degrades to a fixed-schedule duty
cycle rather than to no protection at all.
| `timeseries/run_overnight.sh:19` | **FIX.** `source venv/bin/activate` → `source "${AMW_VENV:-$HOME/venvs/amw}/bin/activate"`. The old line could not work after §1.4. |

**Why one period per invocation.** `panel.py:155` accumulates a period's rows in memory and
`panel.py:236` writes the parquet only at the end. A period boundary is the only point
where stopping costs nothing (§5.2). Cooling there is free; interrupting mid-period discards
up to 28 minutes of inference.

**Failure modes designed for, not assumed away:**

- `pmset -g therm` missing or output unparseable → probe returns empty, nominal cooldown
  only, run continues. Never blocks on a measurement it cannot take.
- `taskpolicy` absent → falls back to `nice -n 10`.
- TF context already initialised → warning, not abort. A warning beats killing a long run.
- torch import failure → silently skipped; it is unused on the `embedding_strategy='none'` path.

**Verified, not assumed:** bash and Python syntax check clean; the `--months` filter
(`run_trials.py:110-113`) accepts quarterly labels — `--months 2025-01,2025-02,2025-03,q2025Q1`
resolves to exactly `['m2025-01','m2025-02','m2025-03','q2025Q1']`, confirmed by replicating
the filter against the generated label set.

**Still NOT implemented** — measure 6, the closed-loop watchdog that `SIGSTOP`s the process
above a temperature limit. It needs sudo `powermetrics`, and suspending mid-period risks the
in-memory panel. The staged cooldown is the safer substitute.

**Not benchmarked.** The thermal effect of these settings on this specific hardware has not
been measured — the reasoning is from where the load provably is (§3.3). `time_tiles.py` can
measure the wall-clock cost of the caps; it cannot measure temperature. Expect roughly
1.5–2× wall-clock per period at `AMW_THREADS=4` on an 8-core machine, plus the cooldowns.

Two non-software points:

- Lid open, plugged in, sustained full load is the worst thermal configuration for a
  laptop. A cooling stand changes more than any code change here.
- 10 h at 100 % CPU and 5 h at 50 % may dissipate similar total energy but at very
  different peak junction temperature. Peak temperature, not total energy, damages hardware.

---

## 4. Threshold provenance — DEEPDIVE

Inventory verified against code; upstream claims verified by reading
`upstream/main:README.md` at commit `bbbcb2d` and `upstream/main:data/outputs/MANIFEST.yaml`.

### 4.1 Complete threshold inventory

**Classifier probability cutoffs**

| file:line | name | value | controls |
|---|---|---|---|
| `core/inference_engine.py:122` | `InferenceConfig.pred_threshold` | **0.5** | dataclass default |
| `core/inference_engine.py:645` | use site | — | `mean_preds > threshold` — strict `>` |
| `core/inference_pipeline.py:105` | `--pred_threshold` | 0.5 | CLI |
| `timeseries/run_trials.py:78` | `--threshold` | **0.4** | fork orchestrator |
| `timeseries/panel.py:109,245` | `pred_threshold` | **0.4** | gates the positives GeoJSON only; the panel stores all probabilities |
| `timeseries/analysis.py:72` | `T_MAIN` | **0.4** | working threshold |
| `timeseries/analysis.py:73` | `T_ALT` | **0.5** | sensitivity |
| `timeseries/analysis.py:74` | `T_GRID` | `arange(0.30, 0.61, 0.05)` | sweep |
| `timeseries/analysis.py:75` | `NEAR_BAND` | **0.05** | fragility metric |
| `core/inference.ipynb` cells 1, 5 | `pred_threshold` | **0.85** | LEGACY, see §6.6 |

**Post-processing — the dual-threshold spatial rule** (`core/postprocess.py`)

| file:line | flag | value | controls |
|---|---|---|---|
| `postprocess.py:244` | `--t-main` | **0.43** | main keep; `confidence >= t_main` (`:68`) |
| `postprocess.py:251` | `--t-iso` | **0.75** | stricter bar for *isolated* patches |
| `postprocess.py:258` | `-k` | **5** | neighbour rank defining isolation |
| `postprocess.py:265` | `-D` | **3.0** | isolation distance, km |
| `postprocess.py:101` | `dissolve_buffer_deg` | **1e-5** | ≈1 m buffered dissolve |

**Cumulative products** (`core/cumulatives_and_diffs.ipynb` cell 1):
`t_main=0.55`, `t_iso=0.8`, `t_andes=0.2`, `k=5`, `D=3`, `min_area_ha=11.0`.

**Cloud / clear-sky**

| file:line | name | value |
|---|---|---|
| `core/gee.py:72` | `DataConfig.clear_threshold` | **0.6** (Cloud Score+ `cs_cdf ≥ 0.6`) |
| `timeseries/coverage.py:80` | `clear_threshold` | **0.6**, enforced identical at `panel.py:152` |
| `timeseries/analysis.py:70` | `TAU_OBSERVED` | **0.95** (chip observed if `clear_frac ≥ 0.95`) |
| `timeseries/analysis.py:71` | `TAU_SENSITIVITY` | **0.50** |
| `_archive/core/collect_sampling_locations.ipynb` cell 21 | training-patch pull | **0.75** ← see §6.9 |

**No NMS / IoU deduplication threshold exists anywhere.** Overlapping chips are merged
geometrically by buffered dissolve. `timeseries/analysis.py:424 geo_iou()` computes IoU
for reporting only, with no cutoff. The only area threshold in the whole pipeline is
`min_area_ha = 11.0` in the cumulative-diff notebook.

### 4.2 Is there a train/val/test split?

**Yes, and it is geographically stratified — this part is methodologically sound.**
Verified in `_archive/core/collect_sampling_locations.ipynb`:

- cell 6: `rng = np.random.default_rng(seed=27)`;
  `gdf['split'] = rng.choice(['train','val'], p=[0.90, 0.10])`
- cell 8: three **geographic holdouts** assigned by spatial join —
  `test1 = geo_holdout_napo-caquetá`, `test2 = geo_holdout_pará-kayapó`,
  `test3 = geo_holdout_caroni`. All three GeoJSONs are in `data/boundaries/`.
- cell 21: a later data batch used 80/20 train/val, same seed.

`core/model_evaluation.ipynb` cell 3 `load_dataset()` *consumes* the split
(`*/{split}/{0,1}/*.tif`); it does not create one.

### 4.3 Are the thresholds *derived* from that split? — No, with one partial exception

**Every threshold value in the repository is a hardcoded literal.** The derivation
machinery exists but is never wired back:

- `model_evaluation.ipynb` cell 7 defines `acc_curve`, `fbeta_curve(beta=1)` and
  `fbeta_curve(beta=0.5)` on the **val split**. Each computes `best = int(np.argmax(...))`
  and writes the optimal `t` **into the plot title only**. Never assigned, never exported.
- cell 8: `threshold = 0.55  # edit` — literal.
- cell 13: `t_main_fixed = 0.43`; `t_iso_grid = np.arange(0.43, 0.90, 0.01)`. This
  **sweeps** `t_iso` and plots isolated-positive recall vs known-negative suppression, but
  performs **no argmax and no automated selection** — the operating point is read off the
  plot by eye.
- cell 15: computes `best = cumul_df.loc[cumul_df.groupby(...)["f1"].idxmax()]` and
  `display()`s it — a derived F1 optimum, again never written back to any default.
- **All notebooks are output-stripped** (verified: every code cell in
  `model_evaluation.ipynb` has `execution_count=None`, zero outputs). There is therefore
  **no stored numerical evidence of any curve optimum anywhere in the repo.**

**The one exception — `t_main = 0.43`.** Upstream's README (commit `bbbcb2d`) states
verbatim:

> "The main confidence cutoff (`t_main = 0.43`) **sits at the peak of the F0.5 curve**, and
> isolated candidate patches must clear a higher bar (`t_iso = 0.75`)."

So 0.43 is *claimed* to be the argmax of the F0.5 curve on the validation split. The code
that would compute it exists (`model_evaluation.ipynb` cell 7). **The claim is not
reproducible from the repository**: training patches are not committed
(`.gitignore: data/training_patches*`), notebooks are output-stripped, and the sidecar
config files that would corroborate it were produced by a script that is not in the repo.

### 4.4 Provenance of the current ensemble's thresholds

`models/48px_v4.10b-18d-20g-21a-22bc-ensemble_config-t0.43.txt:1` records
`Training dataset: training_patches2026-05-04T09:47` and average precisions
`{'val': 0.9911, 'test1': 0.9899, 'test2': 0.9959, 'test3': 0.9889,
'combined-val-test2-test3': 0.9896}`, followed by per-split classification reports at
0.43. Support counts (`test3`: 665 no-mine / 343 mine; combined: 3,156 / 723) match
upstream's README figures for the "Venezuela holdout" exactly, confirming
`test3 = geo_holdout_caroni = Venezuela`.

**Flag: the generator of these two files is not in the repository.**
`model_evaluation.ipynb` cell 9 and `_archive/core/train_model.ipynb` cell 25 write a
*different* format (plain-text `classification_report`, no `Avg precisions` line, no
`combined-…` key, not JSON-dumped). Grepping for `Avg precisions` across all
`.py`/`.ipynb`/`.md` in fork and upstream returns **no matches**.

### 4.5 Verdict per threshold

| threshold | status |
|---|---|
| `t_main = 0.43` | **DERIVED (claimed)** — peak of F0.5 curve on val, per upstream README. Not reproducible from repo. |
| `t_iso = 0.75` | **JUDGMENT**, informed by the sweep in `model_evaluation.ipynb` cell 13. No stated optimality criterion. |
| `0.55 / 0.8` (cumulative) | **JUDGMENT**. Justified qualitatively ("errors accumulate year on year"), metrics reported post hoc. |
| `0.4` (raw inference) | **OPERATIONAL**, not statistical. Set below `t_main` so marginal detections stay on disk and post-processing can be re-cut without re-running inference. Upstream MANIFEST names it `raw_min_confidence`. |
| `0.2` (Andes supplemental) | **JUDGMENT**, to catch very small-scale mining in partner regions. |
| `k = 5`, `D = 3 km` | **NOT DETERMINABLE FROM AVAILABLE SOURCES.** No source in fork or upstream explains these. |
| `clear_threshold = 0.6` | **NOT DETERMINABLE.** `docs/DECISION_REGISTER.md:54` calls it "Google guidance, 0.50–0.65" but cites no document. |
| `TAU_OBSERVED = 0.95` | fork-set, `analysis.py:70`, no derivation. Ours to justify. |

**Bottom line:** the only threshold in the entire pipeline with a stated derivation is
`t_main = 0.43`. Everything else is a hardcoded constant justified by prose judgment or
not justified at all. The evaluation notebook's `argmax` results are displayed and
discarded; no threshold is written back into any config, constant, or default by code.
`docs/DECISION_REGISTER.md:158` already flags "H1 τ and E3 threshold" as **open** items to
"set from sensitivity sections, then freeze" — still outstanding.

### 4.6 What we would have to do to derive our own threshold

We cannot re-derive upstream's operating point: the training patches are not in the repo
and `training_patches2026-05-04T09:47` is not obtainable from here. What we *can* do,
entirely within our own data:

1. The panels already store **every** chip probability, so any threshold is a post-hoc
   choice — `analysis.py:74 T_GRID` already sweeps 0.30–0.60.
2. `analysis.py:75 NEAR_BAND = 0.05` measures how many chips sit within ±0.05 of the
   threshold — the **fragility** of the monthly-vs-quarterly comparison to the threshold
   choice. If that fraction is large, the cadence result is threshold-dependent and must
   be reported, not hidden by picking one number.
3. Defensible framing for the paper: report the cadence comparison across the whole
   `T_GRID`, not at a single τ, and state explicitly that 0.4 is inherited from upstream's
   `raw_min_confidence` and not optimised on our data.

---

## 5. Parallelism, intermediate outputs, caching, failure recovery

### 5.1 Concurrency model — three nested levels, one knob

```
run_trials.py         SEQUENTIAL over 16 periods            (:122 for spec in specs)
 └ panel.py           SEQUENTIAL over batches of max_workers tiles   (:161)
    ├ fetch phase     ThreadPoolExecutor(max_workers)  — 2×N concurrent EE requests (:165-171)
    │                   N imagery futures  (_fetch_pixels_safe)
    │                   N coverage futures (_fetch_cov_safe)
    └ inference phase SERIAL, per tile, under self._tf_model_lock   (:192, engine :619)
```

`--max_workers` (default 8) controls **both** the thread-pool size and the batch size —
`panel.py:161` uses `max_workers` as the batch stride. One knob, two effects. There is no
process-level parallelism; everything is one Python process, one TF session.

`gee.py:223-280 get_tile_data_concurrent` also implements a thread pool with its own retry
pass — but **`panel.py` does not use it**; it calls `get_tile_data` per tile from its own
pool (`panel.py:259`). So `gee.py`'s two-pass retry logic is dead code on the timeseries
path.

### 5.2 Intermediate outputs — what lands on disk, when

| artifact | written | granularity | survives a crash? |
|---|---|---|---|
| `cache/<year>/S2L1C_<tile>_<start>_<end>.tif` | during fetch, `gee.py:215-219` | **per tile** | **yes** |
| `cache/<year>/S2COV_<tile>_<start>_<end>.tif` | during fetch, `coverage.py:179-183` | **per tile** | **yes** |
| `panels/panel_<label>.parquet` | `panel.py:236`, **end of period only** | per period | **no** |
| `tiles/tile_stats_<label>.csv` | `panel.py:240`, end of period | per period | no |
| `detections/detections_<label>.geojson` | `panel.py:247`, end of period, only if ≥1 positive | per period | no |
| `runlog_<stamp>.log` | streamed, `run_trials.py:101` | continuous | yes |

**This is the critical asymmetry.** The tile cache is written incrementally, per tile. The
panel is accumulated **in memory** (`rows: List[dict]`, `panel.py:155`) for the whole
period and written once at the end. If the process dies at tile 79 of 80, **the parquet is
lost** — but the 79 cached `.tif`s survive, so the retry is nearly free (network fetch
skipped, only CNN inference repeats: ~7.5 s × 80 ≈ 10 min instead of ~12 min).

Memory held per period: 42,320 rows × ~10 fields + Shapely geometry ≈ tens of MB. Not a
constraint at this AOI size; it would become one at basin scale.

### 5.3 Resume logic — file-level, both layers

- **Period level**, `run_trials.py:123-126`: if `panel_<label>.parquet` exists and
  `--force` is not set, the period is skipped entirely. Verified in
  `runlog_20260728_223615.log` lines 2-6 ("panel exists, skipping (resume mode)").
- **Tile level**, `gee.py:180-188` and `coverage.py:156-161`: if the `.tif` exists it is
  loaded from disk; on a `rasterio` exception the cached file is treated as corrupt and
  refetched. Failure to *write* the cache is logged as a warning and does not abort
  (`gee.py:217-219`).

**Consequence:** re-running the identical command after any crash is always safe and always
cheap. That is a genuine strength of this pipeline.

### 5.4 Failure handling — three layers, verified

1. **Per-request**, `gee.py:166` `@retry.Retry(timeout=240)` from `google.api_core` on
   `get_tile_data`, plus `googleapiclient`'s own 5-attempt backoff (visible in the run-1
   log: "Sleeping 1.96 seconds before retry 1 of 5").
2. **Per-tile**, `panel.py:257-270`: `_fetch_pixels_safe` / `_fetch_cov_safe` never raise;
   they return `None`. `panel.py:177-185` records the tile as `status=FAILED` in the stats
   CSV and **continues**. The panel simply has no rows for that tile — the analysis treats
   it as missing, never as zero. Correct behaviour for the missingness question.
3. **Per-period**, `run_trials.py:136-140`: any exception is logged with `log.exception`,
   the label appended to `failures`, loop continues. Exit code 1 if any period failed;
   re-running retries only those.

**What is NOT handled:** a hard kill (`SIGKILL`, OOM, thermal shutdown, power loss) loses
the in-flight period's parquet. There is no checkpoint below period granularity.

### 5.5 Caching — lifetime, size, limits

- **Location:** `data/timeseries/cache/<year>/` (`run_trials.py:92`).
- **Key:** `{collection}_{tile.key}_{start}_{end}.tif` (`gee.py:178`,
  `coverage.py:125-126`). The key encodes the **date window**, not the cadence — so a
  monthly and a quarterly window over the same tile are different files, and **there is
  zero cache reuse between the monthly and quarterly runs.** That is why quarterly periods
  cost ~28 min against ~12 min for monthly: separate fetches *and* a composite spanning 3×
  the scenes.
- **Lifetime: unlimited.** No TTL, no LRU, no size cap, no eviction, no invalidation.
  Nothing ever deletes a cache file. It persists until you `rm` it.
- **Not versioned by parameters.** The key does not include `clear_threshold`. If you
  change `DataConfig.clear_threshold` from 0.6, **stale tiles built at 0.6 are silently
  reused.** A real correctness hazard. Workaround: change `--cache_dir` whenever you change
  composite parameters.
- **Size, measured:** 4.8 GB per year at this AOI (2,560 files). Gitignored
  (`.gitignore`: `data/timeseries/cache/`).
- **Server side:** Earth Engine's own `computePixels` results are not cached by us; EE may
  cache internally but that is not observable or controllable from here.

### 5.6 Measured runtime from run 1 (plan against this, not the README)

From `data/timeseries/runlog_*.log`, `la_pampa_corridor`, 80 tiles, `max_workers=8`:

| period type | measured | n |
|---|---|---|
| monthly | 9.9 – 13.8 min (median ≈ 12.0) | 11 |
| monthly (m2025-12) | 26.0 min | 1 |
| quarterly | 27.2 – 29.2 min | 4 |
| **m2025-11 (outlier)** | **344.6 min = 5.7 h** | 1 |

Sum of period times: **603 min = 10.1 h**. Excluding the m2025-11 outlier: **259 min =
4.3 h** — which is what `timeseries/README.md:39` predicts ("roughly 4–7 h"). The README
estimate is correct for the healthy case; the 10 h wall clock came from one degraded period.

**The m2025-11 stall is diagnosable and was not a CPU problem.** Batch timings from
`runlog_20260728_223615.log`:

```
23:37:04  tiles  1-8/80
01:29:28  tiles  9-16/80   ← 112 min for 8 tiles
03:58:45  tiles 17-24/80   ← 149 min for 8 tiles
04:54:32  tiles 25-32/80   ←  56 min
05:04:14  tiles 33-40/80   ←  10 min   (recovering)
05:06:51  tiles 41-48/80   ←   2.6 min (normal)
… remaining batches 2.5-2.7 min each
```

The same log holds 12 `googleapiclient` retry warnings, all `SSLEOFError` /
`BrokenPipeError` against `oauth2.googleapis.com` and
`earthengine-highvolume.googleapis.com`. **Network / token-refresh degradation, not
compute.** During those 5 hours the CPU was mostly idle on sockets — so this incident is
*not* the thermal event; but wall-clock estimates are unreliable and an unattended run can
hang for hours without failing.

**Missing safeguard:** there is no per-tile or per-period timeout. A hung EE request blocks
a worker thread indefinitely — `google.api_core.retry` caps a single call at 240 s but
outer retries are unbounded. A wall-clock budget per period (kill and mark FAILED after N
minutes) does not exist and is worth adding.

---

## 6. Documentation contradicted by code

Listed because the docs have not been maintained. **Follow the code, not these.**

1. **`docs/DECISION_REGISTER.md:96-97` names flags that do not exist.** It says G1 is
   `postprocess.py --threshold` and G2 `--low_area_conf_threshold`. The actual flags
   (`core/postprocess.py:244, 251`) are `--t-main` and `--t-iso`. The documented commands
   fail.
2. **`docs/CODE_WALKTHROUGH.md:539` gives an unrunnable command line**
   (`--threshold 0.925 --dissolve --low_area_conf_threshold 0.975`). None of
   `--threshold`, `--low_area_conf_threshold`, `--ndvi_threshold` (`:553`) exist in the
   current `postprocess.py`. The doc's own banner (`:4-18`) marks these legacy — disclosed
   but not fixed.
3. **`docs/CODE_WALKTHROUGH.md:224` describes a size-conditional threshold**
   ("0.925, +0.975 for polygons < 30 ha"). No size-conditional rule exists in current
   code — it was replaced by the *spatial-isolation*-conditional rule.
4. **Inconsistent comparison operators.** `inference_engine.py:645` uses
   `mean_preds > pred_threshold`; `postprocess.py:68` uses `confidence >= t_main`. A chip
   at exactly the threshold is dropped by one and kept by the other.
5. **`README.md:8` is wrong.** It claims "all member `.h5` files are also in `models/`".
   `models/` contains only the ensemble `.h5` and its two config `.txt`s. The members are
   in `_archive/models/`. `README.md:26-27` contradicts line 8 two paragraphs later.
6. **`core/inference.ipynb` still runs the legacy model at `pred_threshold = 0.85`** and
   filters at 0.85 — meaningless for the current v4.10b ensemble.
7. **`model_evaluation.ipynb` cell 15's `t_iso_offset = 0.25` rule does not reproduce the
   published relaxed pair.** It gives 0.55→0.80 (correct for the stringent product) but
   0.43→0.68, not the published 0.75.
8. **`DECISION_REGISTER.md:74` claims config files for "t0.39/0.4/0.43/0.5".** For the
   *current* ensemble only t0.4 and t0.43 exist. t0.39/t0.5 belong to the superseded
   `48px_v4.10b-18d-20cg-21a-ensemble` (note `20cg`, not `20g`) in `_archive/models/`.
9. **Train/inference preprocessing mismatch, undocumented anywhere.** Training patches were
   pulled with `clear_threshold = 0.75`
   (`_archive/core/collect_sampling_locations.ipynb` cell 21); inference uses 0.6
   (`gee.py:72`). The model sees composites built from a different clear-sky criterion than
   it was trained on. Whether this matters is untested.
10. **`timeseries/README.md:18` says "~70 tiles"; the actual count is 80**
    (`runlog_20260728_223615.log:7`, and 2,560 cache files = 16 × 80 × 2).
11. **`timeseries/README.md:24` and `run_overnight.sh:5` give the wrong repo path**
    (`~/RA_Raji/…` instead of `~/Documents/RA_Raji/…`). This is the same stale path that
    broke the venv (§1.1). Run-1 logs confirm the old location.
12. **`run_overnight.sh:36` hardcodes `--year 2025`** with no override. For run 2 you must
    call `run_trials.py` directly or edit the script.
13. **`README.md:51` says "`source venv/bin/activate` (deps pinned in
    `requirements-lock_2026-07-17.txt`)"** but gives no install command, so a fresh machine
    has no documented path to a working environment. §1.6 above is that path.
