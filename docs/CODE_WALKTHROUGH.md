<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# The Complete Code Walkthrough

> ## ✅ PART 0 REWRITTEN 2026-08-04
>
> Part 0 (the setup manual) was rebuilt from scratch after a full day of setup
> failures on a new laptop. Every command in it is verified against the installed
> library source or against an observed error. The incident log with root-cause
> evidence for each fault is `docs/RUN2_SETUP_AND_AUDIT.md` §1.
>
> **The three things that changed and will surprise you:** the venv now lives at
> `~/venvs/amw`, **outside** the repo; Earth Engine auth needs
> `auth_mode='localhost'`; and the tile cache must be kept out of iCloud.

> ## ⚠️ STATUS (2026-07-27): parts of this document describe the LEGACY model
>
> This walkthrough was written against the **old SSL4EO-MLP pipeline**
> (`48px_v0.X-SSL4EO-MLPensemble`, folder `gee/`). Upstream has since merged the
> July 2026 rebuild (PR #29): the production model is now the **from-scratch CNN
> ensemble `48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`** run with
> `embedding_strategy='none'` from folder **`core/`**. Consequences:
>
> - **Authoritative doc for the new model:** `pipeline_spatial_card.md`. Where
>   this walkthrough and the card disagree, the card wins.
> - **Legacy-only sections** (accurate for the archived pipeline, NOT the current
>   one): 0.5 (SSL4EO backbone download — **not needed anymore**), Part I "two-model
>   sandwich" & model history table's "current" row, Part III models-on-disk
>   (v0.X classifier + backbone), any command using `--pred_threshold 0.85`,
>   paths under `data/outputs/48px_v0.X-...`, and Part XV-B's 0.925 cut.
> - **Folder change:** the old `gee/` folder was deleted upstream and is archived
>   here at `_archive/gee_pre-core_2026-07-27/`. All commands now run from
>   **`core/`** (updated in 0.6/0.8 below).
> - **Still correct** (mechanics unchanged): tiling/chipping geometry (Part V),
>   GEE compositing & cloud masking (Part VI), file formats, most traps (Part XIV).
> - **New in this fork:** `timeseries/` (cloud-coverage flag + monthly-vs-quarterly
>   trial machinery; see `timeseries/README.md`).

**AMW Mining Detector — research fork.** This document explains the entire live codebase in execution order: every file, every function, every package, every file format, and every quirk. It is written to be read offline, top to bottom. The same knowledge is embedded as comments inside the scripts and notebooks themselves, so you can read code and document side by side.

**The 30-minute path:** read Parts I–III (big picture, packages, models), then Part IV–VII (the run, start to finish), then skim Part XIV (traps). Deep-dive the rest as needed. **Setting up a new machine:** Part 0 below is a complete zero-to-running manual.

---

## Part 0 — The Setup Manual (new laptop → running inference)

Every command explained. Written for macOS (the project machine); Linux deltas noted where they matter. Budget ~10 GB disk (code + venv + models + caches) and one Google account.

### 0.1 System tools: Homebrew, Git, Python

macOS ships without a package manager. Install **Homebrew** (the de-facto one — it installs software into `/opt/homebrew` without touching the system):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

(`curl -fsSL <url>` downloads a file — `-f` fail on server errors, `-s` silent, `-S` still show errors, `-L` follow redirects; piping into `bash -c` executes it as a script.) Then:

```bash
brew install git python@3.11
python3.11 --version    # expect Python 3.11.x
git --version
```

**Why 3.11 is not a preference but a hard ceiling:** `tensorflow==2.14.0` publishes no wheels above cp311. Python 3.12, 3.13 and 3.14 are all impossible. If a dependency resolver ever tells you `pandas==3.0.3 depends on numpy>=2.3.3`, it is not a bad lock file — it means you targeted the wrong interpreter, because pandas raises its numpy floor on newer Pythons. Linux: `sudo apt install git python3.11 python3.11-venv` (deadsnakes PPA on older Ubuntu).

### 0.2 Get the code

```bash
cd ~/Documents/RA_Raji                         # or wherever you keep projects
git clone https://github.com/<your-account>/AMW-mining-detector.git
cd AMW-mining-detector
```

`git clone` copies the full repository including history. No Git-LFS is involved — the classifier `.h5` ships inside the repo; only the legacy 700 MB backbone would need fetching separately (§0.5 — not needed for the current model).

> **Do not move or rename the repo after building the environment.** Virtualenvs hardcode
> the absolute repo path in `bin/activate` and in every console-script shebang. Moving the
> folder leaves `activate` prepending a **non-existent** directory to `PATH`: the shell
> still prints `(venv)`, but `python` silently falls through to some other interpreter and
> every import fails. This cost a full day once. See `RUN2_SETUP_AND_AUDIT.md` §1.1.

### 0.3 The Python environment — **build it OUTSIDE the repo**

```bash
# 1. locate a 3.11 interpreter (uv is optional; brew's python3.11 works identically)
UVPY=$(uv python find 3.11)     # or: UVPY=$(brew --prefix python@3.11)/bin/python3.11
"$UVPY" -VV                     # must print 3.11.x

# 2. create the venv in a plain local directory, NOT in the repo
mkdir -p ~/venvs
"$UVPY" -m venv ~/venvs/amw

# 3. install the pinned stack
source ~/venvs/amw/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/RA_Raji/AMW-mining-detector/requirements-lock_2026-07-17.txt
```

**Why outside the repo.** Originally a hard constraint: the project lived under
`~/Documents`, which macOS syncs through iCloud, and that file provider refuses the
symlink and file-flag operations venv creation performs — `python -m venv venv` returned
`Error: [Errno 1] Operation not permitted`, unfixable with `chmod` or `sudo`
(`RUN2_SETUP_AND_AUDIT.md` §1.4).

**The repo moved to `~/RA_Raji/AMW-mining-detector` on 2026-08-07, so this is no longer
forced** — but keep the venv out anyway. It costs nothing and it makes the environment
immune to any future repo move, which is what caused the original day-long failure (§0.2).

**Never put the project back under `~/Documents`.** Beyond the venv problem, iCloud evicts
cold files to `.icloud` placeholders, and a placeholder is not something `rasterio` can
open — with a 4.8 GB/year tile cache that is a live hazard, not a theoretical one.

**Concepts.** A **venv** is a private copy of Python + packages, so this project's pinned
versions cannot conflict with anything else on the machine. `source .../activate` modifies
the current shell's `PATH` so `python`/`pip` point into it — re-run it in every new
terminal. The **lock file** pins every package to the exact working version
(`tensorflow==2.14.0`, `torch==2.13.0`, `geopandas==1.1.0`, `earthengine-api==1.7.35`,
`descarteslabs==2.1.2`, …) — always prefer it to the loose `requirements.txt`. Expect a
3–4 GB download. **Linux note:** the lock was generated on macOS and includes
`tensorflow-macos==2.14.0`; on Linux delete that line from a copy and install from the copy.

**Do not run `uv init`, `uv sync` or `uv add` in this repo.** It is a `requirements.txt`
project, not a uv project. `uv init` writes a `pyproject.toml` pinned to `>=3.13` with
`dependencies = []`, a `.python-version` of `3.13`, and a *second* venv at `.venv/` — after
which nothing resolves. The only uv command that belongs here is
`uv pip install -r requirements-lock_2026-07-17.txt`, and even that is optional.

### 0.3b Verification gate — run this every session, before anything else

```bash
python -VV                 # must be 3.11.x
which python               # must be /Users/<you>/venvs/amw/bin/python
python -c "import ee, geopandas, tensorflow, numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

Expect `3.11.15`, the `~/venvs/amw` path, and `1.26.4 3.0.3`.

**`(venv)` in the prompt proves nothing.** If `rye` is installed, `~/.rye/shims` sits ahead
of the venv on `PATH` and intercepts bare `python`. A venv with a valid path still wins
(activation prepends its `bin/`), but a venv with a dead path falls straight through to the
shim — producing `ModuleNotFoundError` for packages that are demonstrably installed. Always
trust `which python`, never the prompt.

### 0.4 Google Earth Engine access (the part people get stuck on)

The pipeline computes cloud-free composites *on Google's servers*; you need (a) a Google Cloud project, (b) that project registered for Earth Engine, (c) local credentials. Four steps:

1. **Create a Cloud project.** [console.cloud.google.com](https://console.cloud.google.com) → project picker (top bar) → *New Project* → any name. Note the **Project ID** (the globally-unique lowercase identifier, e.g. `amw-research-2026` — ID, not display name; you'll need it everywhere).
2. **Register it for Earth Engine.** [code.earthengine.google.com/register](https://code.earthengine.google.com/register) → choose the project → *Noncommercial / research* (free). Also enable the API: console → *APIs & Services → Library →* "Google Earth Engine API" → *Enable*.
3. **Authenticate — use `localhost` mode, not the default.**

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

   **Why not bare `ee.Authenticate()`.** `oauth.py:550-558` auto-selects the mode; because
   `_no_gcloud()` (`oauth.py:397`) is false when gcloud is installed, it falls through to
   `auth_mode='gcloud'`, which shells out using **gcloud's own OAuth client ID**
   (`764086051850-…`). Google now blocks the `drive` scope for that client ID, so the
   browser returns *Access blocked*. `earthengine-api` ships its own client ID
   (`517222506229-…`, `oauth.py:43-45`), which the `localhost` and `notebook` modes use.
   Forcing `localhost` bypasses gcloud entirely. Port defaults to 8085 (`oauth.py:607`);
   use `auth_mode='localhost:0'` to auto-select if that port is taken.

   **Dropping `drive` and `devstorage` is safe — verified, not assumed.** `core/gee.py:55`
   requests exactly `["https://www.googleapis.com/auth/earthengine"]` on the
   service-account path, and all data flows through `ee.data.computePixels`; nothing
   touches Drive or Cloud Storage. Reduced scopes also do not break `ee.Initialize()`:
   `data.py:237` calls `oauth.is_valid_credentials(credentials)`, whose signature
   (`oauth.py:154`) takes **no** `scopes` argument and only attempts a token refresh. The
   scope check lives solely in `_valid_credentials_exist()` (`oauth.py:139-150`), used only
   by `authenticate()` to decide whether to skip the flow.

   Credentials land in `~/.config/earthengine/credentials`. Verify with
   `ls -l ~/.config/earthengine/credentials`.

   Avoid `auth_mode='notebook'` for unattended runs — `oauth.py:510` states those
   credentials **expire in 7 days**.

4. **Optional: the gcloud CLI.** Only needed for `gsutil` (§0.7) and ADC. `ee.Initialize()`
   reads `~/.config/earthengine/credentials` first and falls back to ADC
   (`data.py:214-231`), so step 3 alone is sufficient.
   ```bash
   brew install --cask google-cloud-sdk     # Linux: https://cloud.google.com/sdk/docs/install
   ```

Finally tell the code which project to attribute requests to:

```bash
export EE_PROJECT=deforestation-and-mining                          # this shell only
echo 'export EE_PROJECT=deforestation-and-mining' >> ~/.zshrc       # every future shell
```

> **`.env` is invisible to Python — this is a silent failure, not a loud one.**
> `core/gee.py:20` reads `os.environ.get("EE_PROJECT", "earthindex")`. No module in this
> repo calls `load_dotenv`, and `python-dotenv` is not installed. Only
> `timeseries/run_staged.sh` parses `.env`, via a `sed` workaround (the file is
> not valid shell — it has spaces around `=`). **Without the `export`, Earth Engine is
> initialised against `earthindex`, upstream's project**, and the resulting permission
> error never names your project.

(`export` sets an **environment variable**, inherited by child processes; `gee.py` reads it at import time. `~/.zshrc` runs at every new zsh shell; on Linux/bash use `~/.bashrc`.)

### 0.5 Model weights

> **NEW MODEL (current):** the CNN ensemble
> `models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` is already in `models/` and
> needs **no additional weights** — no SSL4EO backbone, no `.pth`, no wrap step.
> Skip the rest of this section unless resurrecting the legacy pipeline.

The Keras classifier is already in `models/`. The SSL4EO backbone is not (700 MB). Download `dino_vit_small_patch16_224.pth` — the DINO ViT-S/16 Sentinel-2 checkpoint from the [SSL4EO-S12 repo](https://github.com/zhu-xlab/SSL4EO-S12)'s pretrained-model links (a ~90 MB timm-style mirror also exists on HuggingFace; the wrap script accepts either) — into `models/SSL4EO/pretrained/`. Then convert it once (repo root, venv active):

```bash
python scripts/wrap_ssl4eo.py models/SSL4EO/pretrained/dino_vit_small_patch16_224.pth
```

This performs the checkpoint surgery of Part III (extract teacher weights, rebuild a 13-channel timm ViT, smoke-test `(1,13,224,224)→(1,384)`) and writes the `dino_vit_small_patch16_224.pt` that `gee.py` hardcodes. If it prints `Smoke test OK` you're done.

### 0.6 Smoke test, then first real run (updated for the new CNN model, `core/`)

```bash
source ~/venvs/amw/bin/activate
cd ~/RA_Raji/AMW-mining-detector
export EE_PROJECT=deforestation-and-mining

cd core
python -c "import gee; print('GEE auth OK')"
```

> **Caveat on this one-liner.** `import gee` does **not** call `ee.Initialize` at import
> time. `gee.py:25 _ensure_earth_engine_initialized()` is a lazy singleton invoked from
> `GEE_Data_Extractor.__init__` (`gee.py:112`). So `import gee` printing OK proves the
> *venv* is correct — it does **not** prove auth works. (Part XIV trap #1 describes the old
> eager behaviour and is stale on this point.) Use the check below to exercise auth for real:

```bash
cd ~/RA_Raji/AMW-mining-detector
python - <<'PY'
import sys; sys.path.insert(0, "core")
import gee, tile_utils, geopandas as gpd
cfg = gee.DataConfig()
ex  = gee.GEE_Data_Extractor("2025-07-01", "2025-08-01", cfg)   # <- this hits Earth Engine
region = gpd.read_file("data/boundaries/la_pampa_corridor.geojson").union_all()
tiles  = tile_utils.create_tiles(region, cfg.tilesize, cfg.pad)
print("tiles in AOI:", len(tiles))                              # expect 80
px = ex.get_tile_data(tiles[0])
print("tile shape:", px.shape)                                  # expect (576, 576, 13)
PY
```

`576 = 552 + 2×12` (`gee.py:68-69`, `gee.py:190`). If this prints, the whole chain —
interpreter, packages, credentials, project, network — is verified end to end.

Then run a small region **with the new model** (no `--embedding_strategy` flag → default `none` → CNN on raw chips):

```bash
python inference_pipeline.py \
    --region_path ../data/boundaries/test_region_small.geojson \
    --model_path ../models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 \
    --start_date 2026-06-01 --end_date 2026-07-01 \
    --pred_threshold 0.4 \
    --image_cache_dir ../data/image_cache
```

Flag meanings: `--region_path` any EPSG:4326 GeoJSON AOI; dates are the composite window, **end-exclusive**; `--pred_threshold` keeps chips with mean-ensemble confidence strictly above it (0.4 = the v4.10b working threshold from its config file; the old 0.85 belonged to the legacy model); `--image_cache_dir` saves every downloaded tile as GeoTIFF keyed by tile+window (re-runs of the same window become free — always set it). Output lands in `data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/…`; progress in `logs/`. Post-process with `postprocess.py` (Part IX), compare periods (Part XV-B).

### 0.7 Optional: SAM2 segmentation stage

Only needed for scar-level area estimation (Part X). From repo root, venv active:

```bash
cd models
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && ./checkpoints/download_ckpts.sh
gsutil cp --billing-project=$EE_PROJECT gs://amazon-mining-watch/sam2/SAM_model_96_px_final.pth .
```

(`pip install -e .` = editable install, importable from anywhere in the venv; `gsutil` ships with gcloud; the fine-tuned weights are 176 MB in a requester-pays bucket — pennies.)

### 0.8 Everyday-use cheat sheet (new model)

```bash
# --- every session, in this order ---
source ~/venvs/amw/bin/activate                                  # venv is OUTSIDE the repo
cd ~/RA_Raji/AMW-mining-detector
export EE_PROJECT=deforestation-and-mining                       # .env is NOT read by Python
which python && python -VV                                       # gate: ~/venvs/amw/..., 3.11.x

# --- monthly/quarterly panels (the fork's pilot; cache kept OUT of iCloud) ---
python timeseries/run_trials.py --out_root data/timeseries_run2 \
    --cache_dir ~/amw_cache/2025 --year 2025 --max_workers 4     # add --limit_tiles 2 --months 2025-07 to smoke-test
python timeseries/analysis.py --panels_dir data/timeseries_run2/panels \
    --tiles_dir data/timeseries_run2/tiles --out_dir data/timeseries_run2/analysis

# --- upstream-style single-period inference ---
cd core
python inference_pipeline.py --region_path ... --start_date ... --end_date ... \
    --model_path ../models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 \
    --pred_threshold 0.4 --image_cache_dir ~/amw_cache/adhoc     # score a region/period
python postprocess.py <detections.geojson> --dissolve --outpath <out.geojson>  # chips → polygons
                                                   # (defaults t_main 0.43 / t_iso 0.75 fit v4.10b;
                                                   #  the old 0.925/0.975 cuts belonged to the legacy model)
python ../utils/gee_data_pull.py <aoi.geojson> --start_date ... --end_date ...  # raw imagery
jupyter lab                                                                # inference.ipynb
```

> **Thermal note.** There is no CPU cap, memory cap, or thermal safeguard anywhere in this
> codebase (audited 2026-08-04; `RUN2_SETUP_AND_AUDIT.md` §3). For long runs prepend
> `OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 nice -n 10` and pass `--max_workers 4`.
> The deleted `run_overnight.sh` called `caffeinate -im`, which *prevents* sleep — a sustained-load
> enabler, not a protection.

> **`run_overnight.sh` was deleted on 2026-08-07.** It ran all 16 periods back-to-back at
> full CPU under `caffeinate -im` — the exact configuration that overheated the laptop —
> hardcoded `--year 2025`, and wrote into run 1's output directory. `run_staged.sh` does
> everything it did, safely. Recover it from history if ever needed:
> `git log --oneline -- timeseries/run_overnight.sh`.

### 0.9 Troubleshooting

Every row below marked **[2026-08-04]** was hit for real during the new-laptop setup. Full
root-cause evidence for each: `RUN2_SETUP_AND_AUDIT.md` §1.

**Environment**

| Symptom | Cause → fix |
|---|---|
| `ModuleNotFoundError: No module named 'ee'` / `'geopandas'` **while `(venv)` shows in the prompt** **[2026-08-04]** | The venv's `activate` hardcodes a stale repo path (repo was moved), so `PATH` gained a dead directory and `python` fell through elsewhere. Diagnose with `which python`. Rebuild per §0.3. **Never move the repo after building the venv.** |
| `zsh: command not found: earthengine` **[2026-08-04]** | Same cause. `earthengine` is a console script inside the venv; it only exists when the venv is genuinely active. Equivalent: `python -c "import ee; ee.Authenticate()"`. |
| `which python` → `~/.rye/shims/python` **[2026-08-04]** | `rye` shims sit ahead of the venv on `PATH`. A *valid* venv still wins after activation; a dead one falls through. Re-activate, or `rye self uninstall`. |
| `Using Python 3.14.0 environment at: ~/.venv` **[2026-08-04]** | A global venv is active and shadowing the project's. `deactivate; unset VIRTUAL_ENV`, then check `~/.zshrc` for an auto-activation line. |
| `Because pandas==3.0.3 depends on numpy>=2.3.3 … unsatisfiable` **[2026-08-04]** | Not a broken lock file — the resolver targeted a Python newer than 3.11, where pandas raises its numpy floor. `tensorflow==2.14.0` has no wheels above cp311. Force 3.11. |
| `warning: … incompatible with the project's Python requirement: >=3.13` **[2026-08-04]** | `uv init` was run and left a `pyproject.toml`. `rm -f pyproject.toml .python-version uv.lock main.py && rm -rf .venv`. |
| `Error: [Errno 1] Operation not permitted` on `mkdir` or `python -m venv` **[2026-08-04, now historical]** | The path is inside iCloud-managed `~/Documents`; the file provider refuses new directories and venv creation. Not fixable with `sudo`. Resolved by relocating the repo to `~/RA_Raji` on 2026-08-07 — recurs immediately if anything moves back under `~/Documents`. |
| `OSError: failed to make path absolute` / `Fatal Python error: error evaluating path` **[2026-08-04]** | Same iCloud cause, seen when a tool spawns the venv's `bin/python` symlink. Same fix. |
| `error: No pyproject.toml found in current directory or any parent` from `uv sync` | Expected — this is a `requirements.txt` project. Do not run `uv sync`/`uv add`/`uv init`. |

**Earth Engine**

| Symptom | Cause → fix |
|---|---|
| Browser returns **Access blocked** during `ee.Authenticate()` **[2026-08-04]** | Auto-selected `auth_mode='gcloud'` uses gcloud's client ID, whose `drive` scope Google blocks. Force `auth_mode='localhost'` and drop the `drive`/`devstorage` scopes (§0.4 step 3). |
| `ee.EEException: Not signed up for Earth Engine` / `no project found` | Project not registered (§0.4 step 2), or `EE_PROJECT` unset — in which case `gee.py:20` silently defaults to upstream's `earthindex`. `export EE_PROJECT=...`. |
| `DefaultCredentialsError` / `Could not automatically determine credentials` | No credentials at all → run §0.4 step 3. |
| `PERMISSION_DENIED … Earth Engine API has not been used` | Enable the API in the Cloud console (§0.4 step 2). |
| Errors that name a project you don't recognise | `.env` is not read by Python. `export EE_PROJECT` explicitly. |
| Many `SSLEOFError` / `BrokenPipeError` retries, batches slowing 40× | Network/token-refresh degradation, not CPU. Observed once for 5 h (`RUN2_SETUP_AND_AUDIT.md` §5.6). No per-period timeout exists — a hung run can stall silently. Lower `--max_workers`, retry off-peak; resume mode makes restarts cheap. |

**Runtime**

| Symptom | Cause → fix |
|---|---|
| `import gee; print('GEE auth OK')` prints OK but a real run fails on auth | That one-liner does **not** test auth — `_ensure_earth_engine_initialized()` is lazy (`gee.py:25`, called at `:112`). Use the end-to-end check in §0.6. |
| `rasterio` errors or long hangs opening a cached `.tif` | The cache is in an iCloud folder and files were evicted to `.icloud` placeholders. Move it: `--cache_dir ~/amw_cache/<year>` (4.8 GB/year, §2.5 of the run-2 doc). |
| `analysis.py` reports more periods than you ran | `analysis.py:94` globs *every* `panel_*.parquet` in the directory with no year filter. Use a separate `--out_root` per run. |
| `FileNotFoundError: .../dino_vit_small_patch16_224.pt` | Legacy path only; the current CNN ensemble needs no backbone (§0.5). |
| `command not found: gcloud` | Restart the terminal after install (PATH), or `brew reinstall --cask google-cloud-sdk`. |
| SSL certificate errors on macOS Python | Run `/Applications/Python 3.11/Install Certificates.command`. |
| TF/Keras errors loading the `.h5` | Wrong TF version → reinstall from the lock file under Python 3.11. |
| Laptop overheats on a long run | No safeguard exists in the code (§3 of the run-2 doc). `OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 nice -n 10 … --max_workers 4`. |

---

## Part 0.5 — WHAT THIS FORK ADDS (fork work vs. upstream AMW)

**Every statement in this Part was mechanically verified against the repository on
2026-08-07 at commit `da7a79d` (upstream/main `bbbcb2d`) — not written from memory.**
Verification commands are given so any claim can be re-checked after the next sync.

### 0.5.1 The one-sentence version

Upstream Amazon Mining Watch detects mining in **single, long composite windows**. This
fork asks whether the same detector can support a **monthly time series** — which requires
distinguishing "no mine" from "no data", so it adds a cloud/coverage measurement layer, a
panel data structure that keeps every chip's continuous score, and the statistical
machinery to compare monthly against quarterly composites. **No upstream inference code was
modified to do it.**

### 0.5.2 `core/` is byte-identical to upstream — verified, all 16 files

```bash
for f in core/*.py core/*.ipynb core/README.md; do
  [ "$(git hash-object "$f")" = "$(git rev-parse upstream/main:"$f")" ] \
    && echo "identical $f" || echo "*** DIFFERS $f"
done
```

All 16 files return `identical`: `gee.py`, `inference_engine.py`, `inference_pipeline.py`,
`model_library.py`, `postprocess.py`, `tile_utils.py`, `tf_darwin.py`,
`dense_embedding_cache.py`, `sam2_build_cog.py`, `sam2_combine_masks.py`, `sam2_mask.py`,
`README.md`, and the four notebooks.

This is a deliberate architectural constraint, not an accident: `git checkout upstream/main
-- core/` must always be safe. The fork reuses upstream's building blocks
(`get_tile_data`, `cut_chips`, `_model_infer`, `_probs_from_predictions`) by *calling* them,
and `panel.py:143-146` asserts those attributes still exist so a future sync fails loudly
rather than drifting silently.

**Consequence for the paper:** every probability this fork reports is bit-identical to what
upstream's `bulk_predict` would produce for the same chip and window. The fork changes what
is *recorded* and *analysed*, never what is *computed*.

### 0.5.3 New files — 100 % fork work

| file | lines | what it does |
|---|---|---|
| `timeseries/coverage.py` | 205 | **The cloud flag.** Per-pixel `N_CLEAR`/`N_TOTAL` scene counts on the same 576×576 grid as the imagery, mirroring `gee.py`'s composite construction exactly. This is what makes "no data" distinguishable from "no mine". |
| `timeseries/panel.py` | 413 | **The panel builder.** One row per chip per period with the *continuous* score, the coverage flag, the six ensemble members, and full provenance. |
| `timeseries/run_trials.py` | 161 | Orchestrator over 12 monthly + 4 quarterly windows; resume-safe. |
| `timeseries/analysis.py` | 723 | The monthly-vs-quarterly statistical comparison → `REPORT.md`, figures, CSV tables. |
| `timeseries/run_staged.sh` | 260 | **The only entry point.** Duty-cycled runner: one period per process, cooldowns, CPU caps, background priority, optional `--analyse`. |
| `timeseries/resource_limits.py` | 140 | CPU thread caps applied before TensorFlow initialises. |
| `timeseries/explore_results.ipynb` | 129 | Interactive exploration of the analysis outputs. |
| `timeseries/README.md` | 62 | Module overview. |
| `time_tiles.py` | 139 | Per-tile timing benchmark for cost extrapolation. |
| `scripts/provenance_report.py` | 140 | Regenerates `docs/PROVENANCE.md`, the per-file fork/upstream manifest. |
| `scripts/sync_upstream.sh` | 51 | Upstream sync with auto-resolution of archived-file conflicts. |
| `requirements-lock_2026-07-17.txt` | 107 | Fully pinned environment (upstream ships only a loose `requirements.txt`). |
| `docs/` × 8 | ~3,500 | `CODE_WALKTHROUGH`, `DECISION_REGISTER`, `GIT_PLAYBOOK`, `LECTURE`, `OPTIMIZATIONS`, `PROVENANCE`, `RUN2_SETUP_AND_AUDIT`, `pipeline_spatial_card`. |

Regenerate this list with:
```bash
git diff --name-status upstream/main HEAD | awk '$1=="A"' \
  | grep -vE "data/|models/|_archive"
```

### 0.5.4 Upstream files modified — only three, and only barely

`git diff --name-status upstream/main HEAD | awk '$1=="M"'` returns five entries, two of
which are project metadata (`.gitignore`, `README.md`). The three code files were checked
by parsing both versions to an AST and comparing executable statements:

| file | executable change |
|---|---|
| `scripts/concatenate.py` | **none** — comments and docstrings only |
| `scripts/dissolve.py` | **one line**: removed a stale hardcoded `default=` on a positional CLI argument |
| `utils/gee_data_pull.py` | **one line**: `sys.path` `'gee'` → `'core'`, because upstream deleted the `gee/` folder |

Everything else in those files is annotation. Re-verify with:

```bash
git show upstream/main:scripts/dissolve.py > /tmp/up.py
python3 -c "import ast;print(ast.unparse(ast.parse(open('scripts/dissolve.py').read())))" > /tmp/a
python3 -c "import ast;print(ast.unparse(ast.parse(open('/tmp/up.py').read())))" > /tmp/b
diff /tmp/b /tmp/a
```

### 0.5.5 The four substantive research contributions

Everything above is inventory. These are the actual intellectual differences.

**(1) The cloud flag — `coverage.py`.** Upstream's pipeline cannot tell a cloud gap from a
true negative. `gee.py:132` masks each scene at `cs_cdf >= 0.6` before the median; a pixel
cloudy in *every* scene survives as fill value 0, and an all-zero chip simply scores "no
mine". For a single annual map that is tolerable. For a monthly time series it is fatal:
apparent dips in mining would just be wet-season cloud, and the missingness is almost
certainly MNAR because clouds *and* mining activity both follow the season.

`coverage.py` fetches a second raster per tile on the identical grid — `N_CLEAR` (scenes
passing the cloud mask) and `N_TOTAL` (scenes covering the pixel) — using
`ImageCollection.count()` on band B2. `N_CLEAR == 0` is exactly "this pixel is fill".
`panel.py:75-88 fill_zero_mismatch` then *empirically verifies* the fill-is-zero assumption
per tile rather than assuming it, and the result is reported in §B of every run.

**(2) The panel instead of a filtered detection list — `panel.py`.** Upstream's
`bulk_predict` writes only chips *above* threshold, cumulatively appended per region. This
fork writes **every** chip, with its continuous probability and its coverage flag. Two
consequences that matter for econometrics: thresholding becomes a post-hoc modelling choice
that can be varied in analysis (`analysis.py:74` sweeps 0.30–0.60), and missingness becomes
an explicit per-chip variable instead of an invisible confound. `chip_id` is deterministic
across periods, so the result is a genuinely balanced panel.

**(3) Ensemble uncertainty — schema v2, `panel.py`.** Verified from the model file: the
production `.h5` is a Keras `Concatenate` of six single-sigmoid CNNs, so `_model_infer`
returns `(n, 6)` and `_probs_from_predictions` (`core/inference_engine.py:632`) collapses it
with `np.mean(axis=1)`. Upstream discards the members. This fork keeps them
(`p_m0 … p_m5`), because *do the six models disagree more in cloudy months?* is a sharper
instrument-stability metric than the mean alone — and it is unrecoverable afterwards without
re-running all inference. `prob` remains bit-exactly `mean(p_m*)`.

**(4) Provenance and reproducibility.** Upstream ships no lock file, no per-file
fork/upstream manifest, and no configuration record in its outputs. This fork adds a pinned
lock file, `scripts/provenance_report.py` + `docs/PROVENANCE.md`, PROVENANCE tags in every
fork-touched file, twelve provenance columns in every panel row, and
`analysis.py:_check_provenance`, which **refuses** to pool panels built with different
`clear_threshold` values and warns on mixed models, schemas or code versions.

### 0.5.6 Engineering added on top (not research, but load-bearing)

- **Resource protection** (`resource_limits.py`, `run_staged.sh`): upstream has no CPU cap,
  memory cap, process priority or thermal safeguard of any kind — audited and confirmed in
  `RUN2_SETUP_AND_AUDIT.md` §3. This fork caps TensorFlow intra/inter-op threads and BLAS
  threads, runs at background QoS, and duty-cycles work against rest.
- **Resume semantics at two levels**: period-level (skip existing parquet) and tile-level
  (disk cache), so any crash costs at most one period's inference.
- **Documented threshold provenance**: `RUN2_SETUP_AND_AUDIT.md` §4 traces every threshold
  in the pipeline to its source, and records that only `t_main = 0.43` has any stated
  derivation — a fact not documented anywhere upstream.

### 0.5.7 What is emphatically NOT fork work

The detector itself. The CNN ensemble, its training data, its architecture, the six member
models, the composite construction, the tiling geometry, the chipping, the dual-threshold
post-processing and the SAM2 segmentation stack are **all upstream Amazon Mining Watch
work**. This fork contributes measurement infrastructure and analysis around that detector.
Any write-up should cite `earthrise-media/mining-detector` for the model and reserve novelty
claims for the cloud flag, the panel design, the ensemble-uncertainty retention, and the
monthly-vs-quarterly comparison.

---

## Part I — The big picture

Strip everything away and the system answers one question, repeatedly: *does this ~480 m × 480 m square of rainforest look like an artisanal gold mine?* It is a **binary patch classifier**, not a segmentation model. The mine *outlines* come later, from an optional second stage (SAM2). A time series is built by asking the same question about the same patches for different date windows.

### The two-model sandwich

```
Sentinel-2 pixels, 48×48 px, 13 bands        ~480 m × 480 m of ground at 10 m/px
        │  ÷10000 → reflectance [0,1]; bicubic resize to 224×224
        ▼
SSL4EO ViT-S/16 backbone  (PyTorch, frozen)  pretrained self-supervised (DINO) on
        │  → 384-number embedding             global Sentinel-2; never trained by AMW
        ▼
Ensemble of 6 MLP heads   (Keras, .h5)       the ONLY part AMW trained; each head:
        │  → 6 sigmoid scores                 384 → 64 → 16 → 1, ~27k weights
        ▼
mean(6 scores) = "confidence" ∈ [0,1];  keep chip if confidence > threshold
```

Why this design? A frozen foundation model already "understands" satellite imagery from pretraining, so AMW needed only ~6,400 labeled examples to train six tiny classification heads. It also explains the odd dual-framework dependency: the 2021–2024 models were pure Keras CNNs, so the classifier infrastructure was already TensorFlow; the 2025 redesign bolted a PyTorch backbone in front of it.

### Model history (why the archive contains what it does)

| Era | Model | Input | Architecture |
|---|---|---|---|
| 2021–22 | 28px v9, 44px v2.x | raw pixels | small Keras CNNs, Descartes Labs imagery |
| 2024 | 48px v3.x + ensembles | raw pixels | Keras CNNs retrained on GEE imagery |
| 2025 (current) | 48px v0.X-SSL4EO-MLPensemble | 384-d embeddings | frozen SSL4EO ViT-S/16 + 6 Keras MLP heads |

### The key numbers

| Quantity | Value | Where defined |
|---|---|---|
| Tile size | 576 px = 5.76 km ≈ 33.2 km² | `DataConfig.tilesize` |
| Pixel resolution | 10 m (Sentinel-2 native for the 10 m bands) | `create_tiles(resolution=10)` |
| Chip size | 48 px = 480 m ≈ 23 ha | `InferenceConfig.geo_chip_size` |
| Stride | 24 px (= 48 // `stride_ratio` 2) | `_resolve_chip_params` |
| Chips per tile | 23 × 23 = 529 (overlapping) | consequence of 576/48/24 |
| Bands | 13 (Sentinel-2 L1C, all bands) | `DataConfig._BAND_IDS` |
| Embedding dim | 384 (ViT-Small) | fixed by backbone |
| Ensemble members | 6 (v0.4, 0.6, 0.8, 0.9, 1.0, 1.3) | inside the .h5 |
| Cloud threshold | cs_cdf ≥ 0.6 | `DataConfig.clear_threshold` |
| Inference threshold | 0.85 (CLI), 0.5 (dataclass default) | `InferenceConfig.pred_threshold` |
| Post-process threshold | 0.925 (+0.975 for polygons < 30 ha) | `postprocess.py` args |

Scale check: the Amazon basin is ~8.5M km² ÷ 33.2 km²/tile ≈ 256k tiles × 529 chips ≈ **135M classifications per basin-wide run** — matching the README's "over 100 million assessments."

### The full data flow

```
region.geojson ──create_tiles──▶ [tile objects]            (geometry only, no pixels)
                                     │ per tile, threaded
GEE servers: S2_HARMONIZED ∩ dates → mask cs_cdf<0.6 → median composite
                                     │ ee.data.computePixels (576×576×13 uint16 DNs)
                    (H,W,13) float32 array
                                     │ ÷10000, clip [0,1]; cut_chips
                    (529,48,48,13) chips + 529 EPSG:4326 squares
                                     │ NHWC→NCHW, bicubic 48→224, ViT (no_grad)
                    (529,384) embeddings ──Keras ensemble──▶ (529,6) sigmoids
                                     │ mean > threshold
                    GeoDataFrame of positive chips (confidence, preds)
                                     │ rewritten to disk after every batch
data/outputs/<model>/<region>_<model>_<thr>_<start>_<end>.geojson
                                     │ postprocess.py: threshold 0.925, dissolve, area
                    merged polygons  │ sam2_mask.py: SAM2 segmentation per tile
                    *-msk.tif / *-logits.tif ──sam2_build_cog.py──▶ COGs + STAC
```

---

## Part II — The cast of packages

Every import in the live code, and why it exists.

**`ee` (earthengine-api).** The Google Earth Engine Python client. Central mental model: you build a *server-side recipe* out of lazy proxy objects (`ee.ImageCollection`, `ee.Image`, filters, `.median()` …) — **no pixels move** until you call an execution endpoint. This repo uses exactly one: `ee.data.computePixels`, which executes a recipe for a small raster and returns it synchronously as a NumPy array. Auth is via Application Default Credentials (`gcloud auth application-default login`) plus a Google Cloud project (env var `EE_PROJECT`).

**`descarteslabs`.** A dead commercial platform whose client library survives here for exactly one class: `descarteslabs.geo.DLTile`, a deterministic global grid of UTM tiles. Nothing else from the package is used; the pinned version (2.1.2) exists only for the grid.

**`geopandas` / `shapely`.** `geopandas.GeoDataFrame` = pandas DataFrame + a `geometry` column of shapely objects + a CRS. Reads/writes GeoJSON, GeoParquet; spatial ops (`buffer`, `union_all`, `sjoin`, `clip`, `explode`, `to_crs`). Shapely provides the geometry objects themselves (`Point`, `Polygon`, `box`). All detections live in GeoDataFrames.

**`rasterio` + `affine`.** Raster I/O (GeoTIFF read/write) and georeferencing. An `affine.Affine` is the 6-number transform mapping pixel (col,row) → map (x,y); `~transform` inverts it (map → pixel). `rasterio.transform.from_bounds` builds one from a bounding box + raster size. `rasterio.features.rasterize` burns vector polygons into a pixel mask (used by the SAM2 spatial prior).

**`rasterstats`.** Imported in gee.py (`zonal_stats`) — a vestige of the removed NDVI masking stage; no live call site remains.

**`numpy` / `pandas`.** Arrays and tables. Note the *structured array* trick in `get_tile_data`: GEE returns one named field per band, restacked via `np.stack([pixels[band] …], axis=-1)`.

**`scipy.ndimage` (`ndi`).** Three functions: `distance_transform_edt` (distance-to-nearest-detection for the SAM2 prior), `gaussian_filter` (logit smoothing), `zoom` (prior resampling).

**`tensorflow` / `keras`.** Loads and runs the .h5 ensemble heads. Inference only (`compile=False`); no training code remains.

**`torch` (+ `torch.nn.functional as F`).** Runs the SSL4EO backbone and SAM2; `F.interpolate` does chip resizing (bicubic) and logit upsampling (bilinear).

**`timm`** (only in `wrap_ssl4eo.py`). PyTorch Image Models — builds the ViT-S/16 skeleton the SSL4EO weights are loaded into.

**`google.api_core.retry`.** The `@retry.Retry(timeout=240)` decorator: exponential-backoff retrying of transient GEE errors, up to 240 s.

**`tqdm`.** Progress bars. **`geojson`** (scripts/concatenate.py only): minimal GeoJSON reader/writer. **`matplotlib`** (tile_utils import): plotting leftover, unused in the live path.

**Standard library workhorses:** `concurrent.futures.ThreadPoolExecutor` (parallel downloads), `queue.Queue` + `threading` (producer–consumer machinery, locks, sentinel events), `dataclasses` (the three config objects), `argparse` (CLIs), `logging.handlers.RotatingFileHandler` (log rotation), `pathlib.Path` everywhere.

Threads, not processes — why that works: all the heavy work happens either in network I/O (GEE downloads) or inside native libraries (TF, torch, GDAL) which release Python's GIL. Threads share memory (cheap queues of NumPy arrays); processes would have to serialize every tile.

---

## Part III — The models on disk

### `models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5` — the classifier

`.h5` = HDF5, a hierarchical container of arrays ("a filesystem inside a file"). Keras stores (a) a JSON architecture description in the `model_config` attribute and (b) weight arrays in groups. The actual architecture inside this file (dumped with h5py):

```
Input (None, 384)
 ├─ Sequential "48px_v0.4SSL4EO": Dense 384→64 relu → Dense 64→16 relu → Dense 16→1 sigmoid
 ├─ Sequential "48px_v0.6SSL4EO": (same)
 ├─ Sequential "48px_v0.8SSL4EO": (same)
 ├─ Sequential "48px_v0.9SSL4EO": (same)
 ├─ Sequential "48px_v1.0SSL4EO": (same)
 └─ Sequential "48px_v1.3SSL4EO": (same)
        ↓ Concatenate → output (None, 6)
```

The six archived individual models still live *inside* this file, wired in parallel — the ensemble .h5 is self-contained. `(None, 384)` means "any batch size × 384 features". Loading (`tf.keras.models.load_model(path, compile=False)`) rebuilds the graph from the JSON — no Python model code needed, which is why archiving `model_library.py` was safe. `compile=False` skips reconstructing optimizer/loss (training-only objects).

The sidecar `_config.txt` records provenance: training dataset filenames and held-out classification reports (e.g. at threshold 0.925: precision 1.00, recall 0.80 for the Mine class on 6,430 samples).

### `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt` — the backbone

PyTorch `.pt`/`.pth` files are Python **pickles**. Two conventions: pickling a `state_dict` (bare tensor dict; safe, portable, needs model code to rebuild) or pickling the **whole `nn.Module`** (architecture + weights in one blob). This repo uses the second — hence the one-line load in gee.py: `torch.load(path, weights_only=False)` (`weights_only=False` is required to unpickle arbitrary objects; unpickling can execute code, so only ever load trusted files).

What you downloaded was the *raw research checkpoint* (`.pth`, 720 MB): a dict with `teacher` and `student` copies of the weights (DINO self-supervised training runs two networks; the teacher is the good one), keys prefixed `module.backbone.` (DDP multi-GPU + SSL wrapper artifacts). `scripts/wrap_ssl4eo.py` performs the surgery: extract teacher, strip prefixes, build a fresh skeleton with

```python
timm.create_model("vit_small_patch16_224", in_chans=13, num_classes=0)
```

— `in_chans=13` rebuilds the patch-embedding convolution for 13 spectral bands instead of RGB's 3; `num_classes=0` removes the classification head so `forward()` returns the 384-d pooled feature vector. It loads the weights, asserts the patch-embed shape is `(384, 13, 16, 16)`, smoke-tests `(1,13,224,224) → (1,384)`, and pickles the whole module (~90 MB: one network, no optimizer state).

At inference the ViT sees a 224×224×13 chip as 14×14 = 196 patches of 16×16 px, runs 12 transformer blocks of width 384, and pools to one 384-vector — the chip's "fingerprint."

### `models/sam2/` — segmentation (optional stage)

The cloned `facebookresearch/sam2` repo (pip-installed in editable mode), its base checkpoint `sam2.1_hiera_small.pt`, and AMW's fine-tuned weights `SAM_model_96_px_final.pth` (Georgia Tech collaboration), overlaid at load time with `load_state_dict(..., strict=False)`. Only imported when `run_sam2` / SAM2 scripts are used.

---

## Part IV — Execution: `inference_pipeline.py`

The command under study:

```bash
cd gee
python inference_pipeline.py \
    --model_path ../models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5 \
    --region_path ../data/boundaries/test_region_small.geojson \
    --pred_threshold 0.85 --start_date 2025-07-01 --end_date 2025-09-30
```

**Stage 0 — import time (a trap).** `import gee` executes gee.py's module level, which calls `ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project=EE_PROJECT)` (gee.py:51). So *importing the module* requires working Google auth and the `EE_PROJECT` env var (default `'earthindex'` is Earth Genome's project — set your own). The `highvolume` URL is GEE's endpoint built for many concurrent programmatic requests. Also at module level: on macOS (`platform.system() == 'Darwin'`), TF is switched to eager execution — a Mac bug workaround; slower, irrelevant for tiny MLPs.

**The configuration pattern.** Three dataclasses in gee.py are the single source of truth: `DataConfig` (what imagery), `InferenceConfig` (which models, how to chip and score), `MaskConfig` (SAM2 paths and prior parameters). Every CLI flag simply mirrors a field; lines 180–194 rebuild each dataclass from parsed args:

```python
config_dict = {f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)}
data_config = gee.DataConfig(**config_dict)
```

Each dataclass has a `__post_init__` doing normalization: `DataConfig` resolves `bands=None` to the collection's band list and creates the image cache dir; `InferenceConfig` validates that if an embedding model is enabled, `embed_model_chip_size`/`embedding_batch_size`/`geo_chip_size` are all set; `MaskConfig` resolves SAM2 checkpoint paths and converts filesystem YAML paths into Hydra config *names* (SAM2's loader uses Hydra `compose`, which wants package-relative names like `configs/sam2.1/sam2.1_hiera_s.yaml`, not paths).

**Logging.** `get_logger` attaches a `RotatingFileHandler` (1 MB per file, 5 backups) writing to `../logs/gee_<region>.log`; every run first logs its full argument dict — your provenance record.

**`main()` (lines 33–56).**

```python
region = gpd.read_file(cli_args.region_path).union_all()   # all features → one geometry
tiles  = tile_utils.create_tiles(region, 576, 0)           # cover it with UTM grid tiles
engine = gee.InferenceEngine(start_date, end_date, data_config, inference_config, mask_config, logger)
preds  = engine.bulk_predict(tiles, region_path.stem)      # everything happens here
```

Then two summary log lines — one of which contains a bug worth knowing: `analyzed_area = len(tiles) * (tilesize/100)**2` is logged as "ha analyzed", but for 576 px tiles the formula yields 33.18 — which is km², not ha (a 5.76 km square is 3,318 ha). Off by ×100, harmless, log-only.

---

## Part V — Geometry: `tile_utils.py`

### GeoJSON and CRS fundamentals

A `.geojson` file is plain JSON: a `FeatureCollection` of `Feature`s, each with a `geometry` (e.g. `Polygon` = list of `[lon, lat]` rings — **longitude first**) and free-form `properties`. Per spec (RFC 7946) coordinates are WGS84 lat/lon = **EPSG:4326**, units of *degrees*. Degrees are not meters: 1° latitude ≈ 111.32 km everywhere, but 1° longitude ≈ 111.32·cos(latitude) km. UTM zones (60 north/south strips, each a local projection in honest meters) are the standard fix; this codebase constantly hops between UTM (tile definitions) and 4326 (everything stored).

### `create_tiles(region, tilesize, pad, resolution=10)`

One call: `DLTile.iter_from_shape(region, tilesize=576, resolution=10, pad=0)` — yields every tile of the fixed global grid intersecting the region. Why a *fixed* grid instead of slicing the bounding box? **Determinism**: tile N of a region is the same square of Earth in every run, every quarter, every script. Caches key on `tile.key`; time series align for free. Each tile carries `key` (unique string ID), `crs` (its UTM zone), `bounds` (UTM meters), `geotrans` (Affine pixel→UTM), and `geometry` — the footprint **reprojected to EPSG:4326** for talking to GeoJSON and GEE.

### `CenteredTile` (lines 9–63)

A DLTile impostor centered *exactly* on a given lat/lon instead of snapped to the grid — used for point-based scoring (`utils/gee_data_pull.py`, training-data collection). Construction: build a throwaway real DLTile at that point just to inherit the correct UTM `crs`; project the point to UTM meters (GeoSeries `to_crs`); take half-width = `tilesize·resolution/2`; snap the upper-left corner to the 10 m grid (`floor(minx/res)·res`, `ceil(maxy/res)·res`); build the standard north-up geotransform

```python
Affine.translation(ulx, uly) * Affine.scale(res, -res)   # y-scale negative: rows grow south
```

then bounds, shape, a synthetic `key` embedding lat/lon/res/size, and the 4326 footprint. `tiles_for_geometry` dispatches: Point → one CenteredTile, MultiPoint → one per point, polygons → `create_tiles`.

### `ensure_tile_shape(raster, height, width=None)` (lines 98–148)

GEE occasionally returns 575 or 577 pixels instead of 576. This center-crops if too large and pads if too small — with `mode="reflect"`, mirroring real imagery instead of stitching on black borders that a classifier could misread as a feature. In: `(H,W,C)`; out: exactly `(height,width,C)`.

### `cut_chips(pixels, bbox, chip_size, stride, crs)` (lines 150–190)

The chipper. Slides a `chip_size` window with `stride` over the array:

```python
for col in range(0, W - chip_size + stride, stride):
    for row in range(0, H - chip_size + stride, stride):
        chip = pixels[row:row+chip_size, col:col+chip_size]
```

The `+ stride` in the bound makes the final window that ends exactly at the edge inclusive: for 576/48/24 → positions 0, 24, …, 528 → 23 per axis → **529 chips**, full coverage, every interior ground point voted on by up to 4 chips. For each chip it computes corner lon/lats by *linear interpolation across the tile's 4326 bounds* (`x_per_pixel = (east-west)/W`) — an equirectangular approximation: pixels are treated as uniform in degrees, so "10 m pixels" is only approximately true (true pixel width = deg·111,320·cos(lat)). Near the equator the error is small but systematic; remember this when doing area math. Returns `(chips array (N,48,48,13), GeoDataFrame of N squares, index-aligned)`. That index alignment is the thread connecting model outputs back to map locations.

---

## Part VI — `gee.py` I: `GEE_Data_Extractor` — data acquisition and ALL the preprocessing

This class owns everything between "I have a tile geometry" and "I have a `(576,576,13)` float32 array."

### `__init__(start_date, end_date, config)`

Stores the dates as strings and immediately calls `_build_composite`. Crucial mental model: `self.composite` is a **lazy server-side recipe** — an `ee.Image` proxy object describing *how* to make the composite. Zero pixels move here. The recipe is built once per engine and reused for every tile.

### `_build_composite` — where the real preprocessing lives (lines 210–255)

For the default `S2L1C` collection:

```python
s2     = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")               # every L1C scene ever
csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")  # per-scene cloud scores
composite = (s2.filterDate(start_date, end_date)                      # keep scenes in window
    .linkCollection(csPlus, ["cs_cdf"])                               # join QA band by scene ID
    .map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.6)))   # mask unclear pixels/scene
    .median())                                                        # per-pixel median over time
```

Line by line: `S2_HARMONIZED` is the Level-1C (top-of-atmosphere) collection with a harmonized DN offset across processing baselines. **Cloud Score+** is Google's learned per-pixel cloud/shadow quality product; its `cs_cdf` band is a calibrated 0–1 "confidence this pixel is clear" (CDF-scaled, so 0.6 ≈ "clearer than 60% of the time"). `linkCollection` attaches each scene's matching QA image by `system:index` — after it, every scene carries a `cs_cdf` band. The `.map(...)` applies `updateMask` per scene: pixels with `cs_cdf < clear_threshold` (0.6) become *masked* (excluded from statistics). Finally `.median()` reduces the whole stack to one image: per pixel, the median of all surviving observations.

The median is the entire compositing philosophy: no mosaic rules, no "best scene" selection — order statistics silently reject leftover clouds (too bright), shadows (too dark), and sensor glitches. Two consequences you must internalize for the research:

1. **Temporal blur.** A pixel's value is the median over the whole window. A mine that appears mid-window is a "half-weight" signal. Shorter windows = sharper time resolution but fewer observations per pixel.
2. **The cloud blind spot.** Nothing records *how many* clear observations survived per pixel. All-cloudy pixel → fully masked → `computePixels` returns **0** → the chip silently contains black holes the classifier still scores. This is precisely what the planned cloud flag fixes.

Other collections: `S2L2A` (surface reflectance, 12 bands — B10 cirrus doesn't exist at L2A) same logic; `S1` (Sentinel-1 radar GRD: filter IW mode + require VV & VH polarizations, `.mosaic()` — radar ignores clouds so no masking); `AlphaEarth` (Google's precomputed 64-band annual embedding product). Unknown → `ValueError`.

### `get_tile_data(tile)` — one tile's pixels (lines 257–312)

Decorated `@retry.Retry(timeout=240)`: any transient GEE failure (rate limit, 500, network) is retried with exponential backoff for up to 240 s before giving up — essential at 250k-tile scale.

Sequence: (1) cache check — if `image_cache_dir` is set and `{collection}_{tile.key}_{start}_{end}.tif` exists, load and return it (this filename scheme *is* the cache key: collection + tile + window). (2) Build `ee.Geometry.Rectangle(tile.geometry.bounds)` — the tile's lat/lon box — and ask the recipe for exactly `tilesize+2·pad` pixels square via `clipToBoundsAndScale(geometry, width, height)`: server-side clip + resample. (3) Execute:

```python
pixels = ee.data.computePixels({
    "bandIds": self.config.bands,        # ["B1",...,"B12"] — 13 bands
    "expression": composite_tile,        # the recipe, clipped
    "fileFormat": "NUMPY_NDARRAY",       # give me numpy, not GeoTIFF bytes
})
```

`computePixels` (2023 API — the reason this repo could migrate off Descartes Labs) synchronously executes the whole recipe *for this tile only* and returns a NumPy **structured array**: one named field per band, each `(576,576)`. The `pixels.dtype.fields` check restacks it to a normal `(576,576,13)` array via `np.stack([pixels[b] for b in bands], axis=-1)`. Values are uint16 **digital numbers = reflectance × 10,000** (masked pixels → 0). Then `ensure_tile_shape` (off-by-one fix), `astype(np.float32)`, optional cache write. Latent bug flagged in the comments: the cache-failure handlers call `self.logger`, which this class never defines — a failing cache read would itself crash with `AttributeError`.

### `save_tile` / `load_tile` — the GeoTIFF conventions (lines 336–403)

`save_tile` writes any `(H,W,B)` array as an EPSG:4326 GeoTIFF. Three product types set dtype/nodata/suffix:

| product_type | dtype | nodata | filename suffix | used for |
|---|---|---|---|---|
| `image` | uint16 (S2) / float32 | — | *(none)* | cached/exported imagery |
| `mask` | uint8 | 2 | `-msk` | SAM2 binary masks (1 mine, 0 not, 2 nodata) |
| `logits` | float32 | NaN | `-logits` | SAM2 log-odds fields |

Mechanics: `np.moveaxis(..., -1, 0)` because rasterio/GDAL store band-major `(B,H,W)`; georeferencing via `from_bounds(tile 4326 bounds, W, H)`; `deflate` compression, internal tiling. `load_tile` is the inverse (read → float32 → back to `(H,W,B)`). `get_tile_data_concurrent` wraps downloads in a `ThreadPoolExecutor(max_workers=8).map` — order-preserving, network-bound, GIL released during I/O.

---

## Part VII — `gee.py` II: `InferenceEngine` — scoring machinery

### `__init__` (lines 408–447)

Order: build the extractor (composite recipe); `tf.keras.models.load_model(model_path, compile=False)` (Part III); create `_tf_model_lock` (Keras `predict` is not thread-safe); pick torch device `cuda > mps > cpu`; unpickle the backbone (`_load_embed_model`: `torch.load(weights_only=False)` → `.to(device)` → `.eval()`, which freezes dropout/norm behavior); construct `SAM2_Masker` only if `run_sam2`.

### `embed(chips, chip_geoms, tile)` (lines 497–555)

The PyTorch half. Cache first: if `embeddings_cache_dir` is set and this tile's `..._embeddings.parquet` exists, read it, drop the geometry column, return `(N,384)` float32 — skipping both download *and* backbone. Else:

```python
tensor = torch.from_numpy(chips).permute(0, 3, 1, 2)        # NHWC → NCHW (torch layout)
tensor = F.interpolate(tensor, (224,224), mode='bicubic')   # 48 → 224 px (ViT's input size)
with torch.no_grad():                                       # inference: no autograd graph
    for i in range(0, len(tensor), 32):                     # batches of 32
        out = self.embed_model(batch.to(device))            # (32, 384)
```

Bicubic because it's the smoothest standard interpolant — the ViT was pretrained on 224×224, so we meet its expected scale. `no_grad` halves memory and speeds things up (no gradient bookkeeping). Results concatenate to `(529,384)`. On a cache miss with caching enabled, embeddings are saved as **GeoParquet** — columns `f0..f383` + chip geometry — georeferenced feature vectors you can re-classify later with a new head *without ever re-downloading imagery* (this is how you'd cheaply retrain heads in your research).

### `_resolve_chip_params`, `_model_infer` (lines 557–572)

Chip size comes from config (48) or, if unset, is inferred from the Keras model's own input shape (ensembles report a list — take the first). `stride = 48 // 2 = 24`. `_model_infer` is just `model.predict` wrapped in the lock.

### `_preds_to_gdf(preds, chip_geoms)` (lines 574–606)

Translates raw model output to georeferenced detections, handling every historical model shape: `(N,1)` sigmoid → squeeze; `(N,2)` softmax → take column 1; `(N,M>2)` — our case, `(529,6)` — ensemble of sigmoids → **row mean** (the code itself notes an M=2 ensemble would be misread as softmax — known ambiguity). Then `idx = mean > pred_threshold` (strict). Empty → empty GeoDataFrame with the right schema (so concjs downstream never break). Otherwise: chip geometries at `idx`, a `confidence` column, and — ensembles only — a `preds` column holding the stringified 6 member scores (stringified because GeoJSON properties must be scalars; keep it — the member *spread* is a free uncertainty measure).

### The producer–consumer machine: `produce_tile_input`, `_consumer`, `bulk_predict` (lines 621–858)

Why this architecture: downloading is slow-but-parallel (network), inference is fast-but-serial (one GPU/model, locks). So: **8 producer threads** fetch tiles concurrently; **1 consumer thread** runs models serially; a bounded queue connects them.

`produce_tile_input(tile)` (producer work): embeddings cache hit → `{"mode": "embeddings", ...}` (fast path; note from the docstring: such tiles *skip inline SAM2*, since no pixels enter the queue); else download → `{"mode": "pixels", pixels, tile}`. Exceptions bubble up to be recorded as failures.

`predict_on_tile_pixels(pixels, tile)` (consumer work): warn if `(tilesize+2·pad) % stride ≠ 0` (coverage gaps); **`normed = np.clip(pixels/10000, 0, 1)`** — the single line converting DNs to [0,1] reflectance, clipping >1 anomalies (bright clouds, specular flashes); `cut_chips`; embed → classify (or legacy direct-CNN path if the backbone is disabled); optional inline SAM2 on positives; returns `(preds_gdf, None)` on success or `(empty, tile)` on failure — the tile-shaped error convention that drives retries.

`bulk_predict(tiles, region_name)` assembles it:

```python
while tries_remaining and retry_tiles:                  # default 2 passes
    for batch in chunks(retry_tiles, 500):              # bound memory per batch
        q = queue.Queue(maxsize=64)                     # BACKPRESSURE: producers block
        # 1 consumer thread + ThreadPoolExecutor(8) of producers
        # as_completed → q.put(item);  producer error → fails.append(tile)
        q.put(sentinel); consumer_done.wait()           # clean shutdown, no deadlock
        predictions = concat(predictions, batch_results)
        predictions.to_file(outpath)                    # FULL rewrite = checkpoint
    retry_tiles = fails; tries_remaining -= 1
```

Details worth owning: the queue's `maxsize=64` caps RAM (≤ 64 in-flight tiles × ~17 MB each); the *sentinel* is a unique `object()` checked by identity — an unmistakable "no more work" token; the consumer sets `consumer_done` in a `finally` so the main thread can never hang; `fails` collects tiles from both producer exceptions and consumer-side prediction errors, and the whole batch loop reruns just those once more. The output GeoJSON is rewritten after every 500-tile batch — a crash costs at most one batch of work. `predict_on_tile(tile)` is the single-tile debug shortcut (notebook cell 3) using the same internals minus threads.

---

## Part VIII — What lands on disk, where, and in what format

Everything a run produces, using a June-2026 `test_region_small` run as the concrete example.

**Detections (the primary product).**
`data/outputs/48px_v0.X-SSL4EO-MLPensemble/test_region_small_48px_v0.X-SSL4EO-MLPensemble_0.85_2026-06-01_2026-06-30.geojson`
Path anatomy: `<inference_output_base>/<model_version>/<region>_<model_version>_<threshold:.2f>_<start>_<end>.geojson`, where `model_version` is the first two `_`-tokens of the .h5 filename and `region` is just the region file's stem (a label — the geometry itself is not stored). Format: GeoJSON FeatureCollection, EPSG:4326. Each feature = one positive **48-px chip square** (~480 m × 480 m) with properties `confidence` (float, mean of the 6 heads) and `preds` (string, the 6 individual head scores). Chips overlap (stride 24), so one mine = several overlapping squares — raw detections always look "blocky and doubled" until dissolved. The file is rewritten after every 500-tile batch during the run (it's safe to peek at it mid-run).

**Logs.** `logs/gee_<region-filename>.log` (rotating, 1 MB × 5): run arguments (provenance!), tile counts, per-batch positive counts, failures.

**Optional caches.** `--image_cache_dir <dir>`: per-tile GeoTIFFs `S2L1C_<tilekey>_<start>_<end>.tif` (uint16 DNs, EPSG:4326, deflate) — set this for your experiments; repeat runs and the cloud-flag work become instant and free. `--embeddings_cache_dir <dir>`: per-tile GeoParquet `..._embeddings.parquet` (529 rows × f0..f383 + geometry) — reusable for retraining heads without any GEE traffic.

**Post-processing output.** Wherever `--outpath` points: a GeoJSON of dissolved polygons with mean `confidence` and (if the size filter ran) `Polygon area (ha)`.

**SAM2 stage outputs.** `data/outputs/sam2/<detections-stem>/`: per-tile `S2L1C_<tilekey>_<start>_<end>-msk.tif` (uint8: 1 mine / 0 not / 2 nodata) and `...-logits.tif` (float32 log-odds, NaN nodata). With `--cog`, a `cog_outputs/` subfolder: merged `mining_mask_*.tif` COGs (grouped by UTM zone + 8° latitude band + date range), one basin-wide mask COG, `tile_index.parquet` (GeoParquet: per-tile bounds + `mine_pixels`/`mine_fraction` stats), `stac_catalog.json` (STAC metadata catalog).

---

hello this is a test


## Part IX — `postprocess.py`: from chips to polygons

Run as: `python postprocess.py <detections.geojson> [more...] --threshold 0.925 --dissolve --low_area_conf_threshold 0.975 --outpath out.geojson`. Steps, in order:

**1. Concatenate + threshold.** All input files are read, concatenated, reprojected to 4326; rows with `confidence < 0.925` are dropped. Two-threshold logic: inference runs cheap at 0.85 so you *keep* marginal detections on disk; analysis re-cuts at 0.925 (the operating point where the held-out precision hits ~1.00 at recall ~0.80). You can re-tune the analysis threshold without re-running inference.

**2. Buffered dissolve — `dissolve_patches` (lines 23–42).** The geometric trick to merge overlapping chip squares into mine polygons:

```python
dissolved_geom = gdf.buffer(1e-5, join_style=2).union_all()   # inflate ~1 m; merge everything
dissolved = ...explode(index_parts=False)                     # split into connected components
dissolved.geometry = dissolved.buffer(-1e-5, join_style=2)    # deflate back
```

Buffer by a hair (1e-5° ≈ 1 m) so squares that merely *touch* become connected; `union_all` fuses the whole layer into one MultiPolygon; `explode` splits it into one row per connected blob; negative buffer undoes the inflation. `join_style=2` (mitred) keeps corners square. Then confidence is re-attached: `sjoin` (spatial join) finds which original chips intersect each dissolved polygon, `groupby("index_right").mean()` averages their confidences.

**3. NDVI mask — DEAD CODE.** The `--ndvi_threshold` branch calls `gee.Masker`, a class that no longer exists (NDVI masking was superseded by SAM2). Using the flag raises `AttributeError`. Don't.

**4. Area + small-polygon filter.** `gdf['Polygon area (ha)'] = gdf.to_crs('epsg:3857').area / 1e4`. **Caveat for your research:** EPSG:3857 (Web Mercator) is not equal-area — it inflates areas by ~sec²(latitude). Within ±5° of the equator the error is <1%, but it is systematic; for real area statistics use an equal-area CRS such as EPSG:6933. `filter_small_polygons` then drops polygons under 30 ha whose confidence is below 0.975 — small detections must clear a higher bar, because isolated single-chip false positives are the dominant error mode.

---

## Part X — The SAM2 stack: from chip polygons to pixel-precise scars

**Why:** chips quantize area to ~23 ha blocks. For honest area estimates AMW fine-tuned SAM2 (Segment Anything 2) to trace actual mine-scar boundaries in RGB renders of the same Sentinel-2 composites.

**`SAM2_Masker` (gee.py:882–1131), per tile:** (1) `polygon_gdf_to_pixel_bbox`: union the detection polygons, take their bounding box, push it through the *inverse* affine (`~transform`) to get pixel coordinates — note the y-flip: geographic NW corner → pixel top-left. This box is SAM2's *prompt* ("segment the thing in here"). (2) `get_rgb`: bands `[3,2,1]` = B4,B3,B2 (true color), `/3000` contrast stretch (reflectance 0.30 → white), uint8 — SAM2 is an RGB model. (3) `predictor.set_image(rgb)`; `predict(box=..., multimask_output=True)` → 3 candidate masks with quality scores; take the best one's **logits** (log-odds field, low-res). (4) `soft_spatial_prior`: rasterize the classifier's detection polygons; `distance_transform_edt` gives each outside pixel its distance to the nearest detection; prior = 0 inside, `−(dist/12)²` outside. Added to SAM2's logits, this quadratically suppresses segments far from where the classifier said "mine" — SAM2 can refine boundaries but not wander off to segment a river. (5) Bilinear-upsample the combined log-odds to tile resolution, Gaussian-smooth (σ=2.5 px), threshold at 0 (log-odds 0 = probability 0.5) → binary mask. (6) `save_tile` both mask and logits as GeoTIFFs.

**`sam2_mask.py`** is the standalone driver (masking after inference, the production mode): reads a detections/cumulative GeoJSON, re-downloads composites per tile (tilesize 256 here, not 576), `bulk_mask_polygons` = simplify polygons for tiling (buffer ±0.005° closes gaps, drop holes) → tile → `gpd.clip` detections per tile (single-threaded — clip isn't thread-safe) → threaded download+segment. **`sam2_build_cog.py`** mosaics thousands of tile masks into a few **COGs** (Cloud-Optimized GeoTIFFs: internally tiled + overview pyramids, so a web client can read any zoom/region via HTTP range requests) using the GDAL CLI chain `gdalbuildvrt` (virtual mosaic — an XML index, no pixel copying) → `gdalwarp` (materialize, align grids, ZSTD) → `gdal_translate -of COG`; groups by UTM zone + 8° lat band + date range; writes the GeoParquet tile index and a STAC catalog. **`sam2_combine_masks.py`** merges two mask COGs with OR logic ("ever mined" cumulation, quarter over quarter) as a *streaming* GDAL VRT derived band whose pixel function is inline Python — output 1 if either input is 1, nodata 2 only where both are nodata.

---

## Part XI — Auxiliary tools

**`core/inference.ipynb`** — the interactive twin of the pipeline (your Phase-1 tool). Cells: (0) imports + autoreload; (1) region/model/dates + the three configs; (2) read region → `create_tiles` → build `InferenceEngine`; (3) `predict_on_tile(tiles[0])` single-tile smoke test; (4) `bulk_predict` full run; (5) `preds.explore()` — folium/Leaflet interactive map of detections right in the notebook.

**`utils/gee_data_pull.py`** — dump raw composite GeoTIFFs for any AOI/collection/date window (no ML). Your best friend for the cloud-flag work: pull a month of imagery and *look* at the masked holes. Note it prepends `../gee` to `sys.path` to import `gee`/`tile_utils`, and importing `gee` triggers the auth-requiring `ee.Initialize`.

**`scripts/concatenate.py`** — GeoJSON merger; rewritten upstream (July 2026) with geopandas — the legacy-`crs` requirement is gone. **`scripts/dissolve.py`** — standalone buffered dissolve (same math as `dissolve_patches`, but drops all attribute columns and writes `<input>-dissolved-<thr>.geojson`; plain `-dissolved` when no threshold — upstream fix 2026-07). **`core/cloud_mask_filter.ipynb`** — reference-only training-era notebook showing how cloudy patches were reviewed (its `viz_tools` import now lives in `_archive/`; restore to run).

---

## Part XII — File-format glossary

**GeoJSON** (`.geojson`): plain-text JSON of features (geometry + properties), always EPSG:4326 lon/lat degrees. Human-readable, git-diffable, slow and fat at scale. The pipeline's vector currency.
**GeoTIFF** (`.tif`): raster (pixel grid) + georeferencing (CRS + affine transform) + dtype + optional nodata, compression, internal tiling. Read/written via rasterio (Python) or GDAL (CLI).
**COG**: a GeoTIFF whose internal layout (tiles + overview pyramids, specific ordering) allows efficient partial reads over HTTP. Same format, stricter arrangement.
**VRT** (`.vrt`): GDAL "virtual raster" — an XML file *describing* a mosaic/computation over other rasters; evaluated lazily, streams without materializing.
**GeoParquet** (`.parquet`): columnar binary table + geometry column. Fast, compact; used for embedding caches and the SAM2 tile index. Read via `gpd.read_parquet`.
**HDF5** (`.h5`): hierarchical array container; Keras' legacy full-model format (architecture JSON + weights).
**PyTorch pickle** (`.pt`/`.pth`): pickled Python objects — here a full `nn.Module` (backbone) and raw checkpoint dicts (SAM2, SSL4EO original). Only load trusted files.
**STAC** (`.json`): SpatioTemporal Asset Catalog — standardized JSON describing geospatial assets (bbox, datetime, asset hrefs). Makes the COGs discoverable by tools.
**Structured NumPy array**: array whose dtype has named fields (per-band) — what `computePixels` returns before restacking.

## Part XIII — Threading, performance, and what to tune

Where time goes in a run: ~all of it is GEE downloads (seconds per tile, 8 in parallel) and the ViT forward pass (529 chips × batches of 32 per tile). The Keras heads are ~free. Levers: `--max_workers` (GEE concurrency; 8 is polite, more risks rate-limit retries), `--max_concurrent_tiles` (batch size between checkpoints), `embedding_batch_size` (GPU memory vs speed), caches (`--image_cache_dir` makes re-runs free; `--embeddings_cache_dir` makes re-*scoring* free). Memory is bounded by design: queue `maxsize=64` × ~17 MB/tile ≈ 1 GB ceiling for in-flight pixels. Threads (not processes) suffice because TF/torch/GDAL/network all release the GIL. On your Mac: backbone on `mps`, heads on CPU (eager mode), SAM2 — CPU only (its device selection skips mps).

## Part XIV — Traps, quirks, and known bugs (memorize this list)

1. `import gee` calls `ee.Initialize` — no auth / no `EE_PROJECT` → import fails. Default project `'earthindex'` is Earth Genome's, not yours.
2. `InferenceConfig.model_path` default is *relative* (`../models/...`) — run scripts from `core/`.
3. `inference_pipeline.py` logs tile area as "ha" but computes km² (×100 off). Log-only.
4. `GEE_Data_Extractor` handlers reference `self.logger`, which is never set → a cache I/O failure raises `AttributeError` instead of a warning. Latent (only bites with `image_cache_dir` set and a corrupt file).
5. `postprocess.py --ndvi_threshold` calls the removed `gee.Masker` → crash. Dead branch.
6. Area in `postprocess.py` uses EPSG:3857 (Mercator): inflated ~sec²(lat). Use EPSG:6933 for science.
7. `_preds_to_gdf` would misread a 2-member ensemble as a softmax model. Ours has 6 — fine.
8. Thresholds are strict `>`; confidence exactly at threshold is dropped.
9. Embeddings-cache-served tiles skip inline SAM2 (documented, by design).
10. Fully-clouded pixels arrive as DN 0 (black holes) and are scored anyway — the cloud-flag motivation.
11. Chip/pixel grids are equirectangular approximations of "10 m" — fine near the equator, systematic with latitude.
12. `SAM2_Masker` picks cuda-else-CPU (no mps): SAM2 is slow on Macs.
13. `concatenate.py` requires the legacy GeoJSON `crs` member; RFC 7946-compliant files raise KeyError.
14. Band *order* is contract: `_BAND_IDS["S2L1C"]` lists B8A before B8 and must match SSL4EO pretraining; never reorder.
15. Cloud Score+ (`cs_cdf ≥ 0.6`) is per-pixel/per-scene masking *before* the median — there is no per-chip cloud metadata afterward.

## Part XV — Where your research plugs in

### A. Granular time series
Nothing in the code assumes yearly/quarterly windows: `--start_date/--end_date` are free. The DLTile grid guarantees the *same tiles and chip geometries* for every window, so time series align exactly by geometry. The physical limit is observation count: Sentinel-2 revisits every ~5 days, so a 1-month window has ~6 scenes before cloud masking — in the wet season possibly zero clear looks per pixel. Granularity in time trades directly against compositing robustness. That is a *measurable* trade — measure it with the cloud flag (C below).

### B. Comparing two periods (June vs July 2026) — the methodology

Run inference twice (same region, same model, same threshold), e.g. windows `2026-06-01..2026-06-30` and `2026-07-01..2026-07-20`. You get two chip GeoJSONs in `data/outputs/48px_v0.X-SSL4EO-MLPensemble/`. Then, per period: confidence-cut (0.925), buffered-dissolve to polygons. Then compare **polygon sets** (never raw chips — stride quantization makes chip-level differencing meaningless):

```python
june, july = (dissolved polygon GeoDataFrames)
gained   = gpd.overlay(july, june, how="difference")      # in July, not in June = new/regrown
lost     = gpd.overlay(june, july, how="difference")      # in June, not in July
stable   = gpd.overlay(june, july, how="intersection")    # persistent mining
# honest areas: equal-area CRS, m² → ha
area_ha  = lambda g: g.to_crs("EPSG:6933").area.sum() / 1e4
```

Report: area(June), area(July), gained, lost, stable, and net = area(July) − area(June) = gained − lost. Interpretation caveats you must carry into any write-up: (1) detection jitter — different composites (clouds! different scene sets) shift confidences, so polygons flicker at the 0.925 boundary; some "change" is noise, not ground change. Mitigations: compare at a lower threshold too (sensitivity analysis), require a minimum change-polygon area (e.g. > 1–5 ha), and check the `preds` member-spread on flickering chips. (2) One month of a 480-m-patch detector resolves *new mine sites* and *large expansions*, not weekly digging. (3) "Lost" mining usually means missed detection (clouds, regrowth reflectance), not restoration — AMW treats mining as cumulative for exactly this reason (the OR-logic in `sam2_combine_masks.py`). For scar-level (not patch-level) change, run the SAM2 stage per period and difference the mask rasters — subtract the uint8 masks and count pixels × (pixel area in an equal-area projection).

### C. The cloud flag — design

The blind spot (Part VI): after `updateMask` + `median`, nobody knows how many clear observations built each pixel. The fix is one extra server-side band. Concept: alongside `.median()`, compute `.count()` on the same masked collection — per pixel, the number of surviving (clear) observations:

```python
masked = (s2.filterDate(start, end).linkCollection(csPlus, [QA])
            .map(lambda img: img.updateMask(img.select(QA).gte(thr))))
composite = masked.median().addBands(masked.select("B4").count().rename("OBS"))
```

Then request 14 bands in `get_tile_data`, split `OBS` off before chipping, and attach per-chip statistics (e.g. `min` and `mean` of OBS over the chip's 48×48 window) as new columns flowing through `chip_geoms` into every detection: each output chip then carries `n_obs_min` / `n_obs_mean` — a per-detection data-quality flag. Downstream: filter or down-weight detections with `n_obs_min == 0` (holes) or low means; report coverage statistics per period so June-vs-July differences can be attributed to data vs ground. This is the guided-edit session: the changes touch `_build_composite` (recipe), `DataConfig` (a flag + band handling), `get_tile_data` (band split), `cut_chips`/`predict_on_tile_pixels` (per-chip stats), `_preds_to_gdf` (carry columns through).

*— End of walkthrough. The same explanations live as comments inside every script and notebook.*
