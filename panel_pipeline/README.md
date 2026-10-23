# panel_pipeline: how to run it, on a laptop and on the Goethe cluster

This folder scores every 48 px chip of a region with AMW's model, month by month and
quarter by quarter, and writes the scores as a chip × period panel. Nothing here
publishes anything; AMW's own driver (`scripts/pipeline.py`) is not used.

The same four commands run everywhere. Part A says what they do and what every option
means, using a laptop as the example. Part B says only what is *different* on Goethe.

```
config  ──►  plan.py  ──►  detect.py (many times)  ──►  status.py  ──►  assemble_period.py
 yaml        once per run   one task = one (frequency, period, subregion)   any time
```

Contents
- [Folder layout](#folder-layout)
- [0. Vocabulary](#0-vocabulary)
- [Part A: on a laptop](#part-a-on-a-laptop)
  - [A1. Environment](#a1-environment)
  - [A2. The config file](#a2-the-config-file)
  - [A3. plan.py](#a3-planpy-once-per-run)
  - [A4. detect.py](#a4-detectpy-the-scoring)
  - [A5. status.py](#a5-statuspy-how-far-is-the-run)
  - [A6. Reading the output](#a6-reading-the-output)
  - [A7. Assembly](#a7-assembly-next-step)
- [Part B: on Goethe](#part-b-on-the-goethe-cluster)
  - [B1. What is different](#b1-what-is-different)
  - [B2. One-time setup](#b2-one-time-setup)
  - [B3. A run on the cluster, step by step](#b3-a-run-on-the-cluster-step-by-step)
  - [B4. The Earth Engine ceiling](#b4-the-earth-engine-ceiling-read-before-planning-a-big-run)
  - [B5. Storage and backup](#b5-storage-and-backup)
  - [B6. Getting the results](#b6-getting-the-results-on-goethe-or-copied-to-a-laptop)
  - [B7. If something breaks](#b7-if-something-breaks)

---

## Folder layout

```
panel_pipeline/
  README.md               this file
  config.yaml             the settings template; copy it to configs/<run>.yaml per run
  plan.py                 step 1, once per run
  detect.py               step 2, the scoring, one task at a time
  status.py               any time: how far a run has got, and the commands still to run
  assemble_period.py      step 3, one period across subregions, plus AMW's postprocess flags
  lib/                    imported by the steps, never run on their own
    config.py periods.py grid.py lock.py provenance.py read_panel.py
                          (exception: `python panel_pipeline/lib/config.py` checks a config)
  checks/                 looking at results and measuring speed; nothing here writes a run
    explore_runs.ipynb  bench_scoring.py  bench_download.py  overlap/ (tile and chip overlaps)
  docs/                   everything written down about the pipeline; start at docs/README.md
    README.md                   which document answers what
    STATE.md                    where the project stands, and every decision with its date
    open_design_decisions.Rmd   decided principles and every open question
    findings.md                 what we measured (test runs, AMW comparison, speed, Earth Engine cost)
    team_brief.md               the short version for colleagues
    code/                       how our scripts work and why they are built that way
    theory/                     computer-science concepts behind the code
    cluster/                    the Goethe plan and the Earth Engine service account
    history/                    older notes, kept for the record, each says what replaced it

amw_comparison/   our scores against AMW's published ones (own README)
cluster/          Goethe job scripts (Part B)
configs/          one committed config per run
requirements/     pinned packages, one list per platform (A1)
core/ scripts/ utils/ docs/   AMW's own code and documents, unchanged; we import four files from core/
```

The steps put `lib/` on Python's search path themselves (one `sys.path.insert` line near
their imports), so every command stays `python panel_pipeline/<step>.py ...` from the
repository root.

---

## 0. Vocabulary

| Word | Meaning here |
|---|---|
| **run** | One `plan.py` result: one folder with one frozen config, one chip list, one model. The whole Amazon is one run; the cluster works *inside* a run. "One run at a time" is a decided principle (`docs/open_design_decisions.Rmd`, P4). |
| **period** | One month (`2024-07`) or one quarter (`2024Q3`). A period runs from its first day to the first day of the next period, which Earth Engine excludes, so every day is imaged. |
| **frequency** | `monthly` or `quarterly`. The two never mix: separate folders, separate panels. |
| **subregion** | One of AMW's six pieces of Amazon_ACA (`data/boundaries/Amazon_ACA/Amazon_ACA_{n}.geojson`). |
| **tile** | AMW's 552 + 2×12 px square at 10 m, about 5.8 km wide. One Earth Engine request per tile and period. |
| **chip** | A 48 px square inside a tile, stride 24 px, so 529 overlapping chips per tile. One row of the panel is one chip in one period. |
| **task** | One `detect.py` command for one (frequency, period, subregion). It writes exactly one folder and nothing else. Tasks are independent, so they can run in any order on any machine. |
| **part file** | `part_NNNNN.parquet`: the scores saved so far by a task. A new one every few minutes. When the task is complete, all parts are merged into `part_00000.parquet`. |

---

## Part A: on a laptop

### A1. Environment

Python 3.11 and the packages in `requirements/`. The pinned versions are the ones in
AMW's `requirements-lock_2026-09-18.txt`, so a score on any machine comes from the
same code.

```bash
# once, from the repository root (uv: https://docs.astral.sh/uv/ ; pip works too)
uv venv venv --python 3.11
source venv/bin/activate
uv pip sync requirements/panel-macos.txt        # on Linux: requirements/panel-linux.txt
earthengine authenticate                        # browser login, once per computer
python -c "import tensorflow, ee, descarteslabs, geopandas, sklearn; print('ok')"
```

- `requirements/panel.in` lists what we need directly; the two `panel-*.txt` files are
  the complete pinned lists generated from it, one per platform. Install the `.txt`,
  never the `.in`. To change a package, edit `panel.in` and regenerate (the command is
  in the file's header).
- The venv folder `venv/` is ignored by git.
- Every command below is run from the repository root, with the venv active.

### A2. The config file

One YAML file holds every setting of a run. `panel_pipeline/config.yaml` is the
template; a real run gets its own copy under `configs/` (committed, so the run is
documented), for example `configs/test_region_2026-10-19.yaml`.

`python panel_pipeline/lib/config.py --config configs/<file>.yaml` checks the file and
prints a summary. It refuses typos, missing keys, keys written twice, wrong types and
impossible values, with a message saying which.

| Section | Keys | Meaning |
|---|---|---|
| `run` | `name`, `output_root` | The run's folder is `<output_root>/<name>`. `name`: letters, digits, `-`, `_`. A relative `output_root` starts at the repository root; `tests_output` is ignored by git. |
| `earth_engine` | `project` | The Google Cloud project ID billed for every request. No default. |
| `periods` | `first_month`, `last_month`, `frequencies` | `2018-01` to `2026-09` gives 105 months and 35 quarters. Quarters must be whole. |
| `region` | `subregion_files`, `subregions`, `andes_boundary`, `clip_to` | `clip_to: null` scores the whole region; a GeoJSON path keeps only the tiles touching that shape (test runs). |
| `model` | `path` | AMW's ensemble `.h5`. Its fingerprint is recorded at plan time; detect refuses a changed file. |
| `imagery` | `collection`, `tilesize`, `pad`, `clear_threshold`, `max_workers` | AMW's `DataConfig`. Only `max_workers` (parallel downloads) can be overridden at run time. |
| `inference` | `stride_ratio`, `tries`, `batch_tiles` | `stride_ratio: 2` gives the 24 px stride. `tries` and `batch_tiles` can be overridden at run time. |
| `thresholds` | `raw`, `andes`, `loose`, `stringent`, `neighbours_k`, `isolation_km` | Used by assembly, not by detect: every chip's score is written whatever its value. |
| `persistence` | `k`, `window_months` | For the later persistence flag. The rule itself is still open (Rmd D8). |

**Frozen:** `plan.py` copies the file into the run (`<run>/run/config.yaml`) and every
later command reads *that copy*, never the live file. So editing `configs/x.yaml` after
planning changes nothing; a different setting means a new run with a new `name`.
Three settings that change only speed, never a score, have command-line overrides
(A4): `max_workers`, `batch_tiles`, `tries`.

### A3. plan.py (once per run)

```bash
python panel_pipeline/plan.py --config configs/test_region_2026-10-19.yaml
```

Builds the run folder, or refuses if it already exists ("a run is planned once").
Prints the config summary, then tile and chip counts, then a `WARNING` if files in the
repository differ from the last commit (commit before a real run, so `provenance.json`
names the exact code).

```
<output_root>/<name>/
├── run/config.yaml        the frozen copy
├── run/provenance.json    git commit, uncommitted files, package versions, host,
│                          sha256 of the model and boundary files, tile and chip counts
├── plan/tiles.csv         one row per (subregion, tile): AMW's six tile lists stacked
└── plan/chips.parquet     one row per chip: chip_id, square, AMW's key, in_region, in_andes
```

Time and size: the 36-tile test region takes seconds; all of Amazon_ACA about 2–3 h and
1.4 GB (274,932 tiles, 145 M chips). Planning needs no Earth Engine.

### A4. detect.py (the scoring)

One command scores one subregion for one or more periods of one frequency, in period
order, writing one folder per period:

```bash
python panel_pipeline/detect.py --run tests_output/test_region_2026-10-19 \
    --frequency quarterly --subregion 2 --from 2026Q1 --to 2026Q3
```

| Option | Required | Meaning | Default |
|---|---|---|---|
| `--run` | yes | the run folder made by plan.py | |
| `--frequency` | yes | `monthly` or `quarterly` | |
| `--subregion` | yes | 1–6; must have tiles in this run | |
| `--from`, `--to` | no | first and last period, like `2018-01` or `2018Q1`; `--from X --to X` is one task | all of the run's periods |
| `--limit N` | no | score only the first N tiles (a test; the task then shows as partly done) | all |
| `--workers N` | no | parallel Earth Engine downloads **in this process** | the run's `imagery.max_workers` |
| `--batch-tiles N` | no | most tiles per part file | the run's `inference.batch_tiles` |
| `--flush-minutes M` | no | save a part at least this often | 2 |
| `--tries N` | no | passes over failed tiles | the run's `inference.tries` |
| `--mac-eager` | no | macOS only: AMW's slower TensorFlow setting, if scoring crashes without it | off |
| `--image-cache DIR` | no | tests only: keep each downloaded image and reuse it (about 9 MB per tile and period; reused images score slightly differently) | none |

What it does, in order: logs in to Earth Engine (fails here if the login is wrong,
before any folder is made); checks the model file against `provenance.json`; loads
the model once; then for each period takes the folder's lock, downloads tiles in
parallel threads, scores each tile as it arrives, and saves what is scored every
`--batch-tiles` tiles or `--flush-minutes` minutes, whichever comes first.

Each task's folder:

```
<run>/detect/<frequency>/period=<tag>/subregion=<n>/
├── part_00001.parquet ...   scores saved so far, one row per chip
├── part_00000.parquet       after completion: all parts merged into this one file
├── _status.json             complete?, tiles planned/saved/failed (with the error text),
│                            chips saved, and one entry per session: when, which
│                            settings and overrides, which code and packages
├── _log.txt                 every progress line, plus each failed or slow tile and
│                            Earth Engine's retries, with timestamps
└── _lock.json               only while a process writes here: host, pid, job, command
```

`period=…/subregion=…` folder names are **Hive partitioning**: readers add them as
columns. Files starting with `_` are skipped by Parquet readers.

**Columns per row:** `chip_id` (a whole number; `chip_id // 1000` is the tile),
`confidence` (AMW's score, float64, mean of the six votes), `vote_1` … `vote_6`
(float32, one per ensemble member, in the order listed in `_status.json`), `n_obs`
and `n_clear` (mean number of images per pixel, all and cloud-free), `empty_share`
(share of pixels with no clear image), and more quality columns: `n_clear_median`,
`thin_share`, `clear_day_mean`, `cs_mean`, `cs_clear_mean`, `ndvi_mean`, `bsi_mean`,
`cirrus_mean`. What each means and how to read them: A6.

**The progress line**, once per save:

```
2026Q3  try 1: 13,282/34,825 tiles saved (+22, 0 failed) | 11 tiles/min, download median 101 s
(slowest 106 s), workers busy 82%, scoring busy 18%, 0 retries | 22 downloading, 22 waiting
for a worker, 0 waiting to be scored, oldest asked for 121 s ago | about 32.9 h left in this try
```

- *workers busy* near 100% and *scoring busy* low: Earth Engine (or the network) is
  the limit; more `--workers` may help, up to the project's limit (B4).
- *scoring busy* near 100%: this computer's scoring is the limit; more workers won't help.
- *retries* counts Earth Engine's "too many requests" and network retries; a rising
  number means you are over a limit.

**Stopping and resuming.** Ctrl-C (or a scheduler's time limit, or a closed terminal)
saves what is scored, gives the lock back and stops; only the downloads under way are
lost. A second Ctrl-C stops at once. **Run the same command again to resume:** saved
tiles are skipped, failed tiles retried, and a finished task ends after a few seconds.
A task that was killed outright (`kill -9`, out of memory, a lost machine) leaves its
lock behind; the next run takes it over after 15 minutes without a heartbeat (at once
on the same machine).

**Two copies at once.** A second detect.py on the same (frequency, subregion) skips
the periods the first one is writing (`SKIPPED: … is being written by …`) and exits
with an error listing them. Two copies on *different* periods or subregions are fine:
that is how the cluster parallelises.

**Exit code.** 0 when every period is complete; 1 after a stop; an error message and
code 1 when periods have failed tiles or were skipped as locked.

### A5. status.py (how far is the run)

```bash
python panel_pipeline/status.py --run tests_output/test_region_2026-10-19
python panel_pipeline/status.py --run ... --frequency quarterly
python panel_pipeline/status.py --run ... --todo > tasks.txt
```

One line per year, one mark per task, subregions in order:
`#` done · `R` running (live lock) · `!` has failed tiles · `~` partly done ·
`S` stopped without cleaning up (stale lock; rerun takes it over) · `.` not started.
Then a list of the tasks worth looking at, with the first error. `--todo` prints the
`detect.py` commands for tasks not done and not running, **period first** (2018-01
subregions 1…6, then 2018-02, …), one per line: this is the task list for the cluster.
It reads only; it is safe while detect.py runs.

### A6. Reading the output

#### The quality columns

Earth Engine first counts **per pixel** (10 m), then detect.py **averages over the
chip's 2,304 pixels** (48 × 48). A pixel's counts are whole numbers; a chip's are
averages, which is why they have decimals.

| Column | Per pixel | Per chip (what is stored) |
|---|---|---|
| `n_obs` | Sentinel-2 images of the period that cover the pixel (cloudy or not) | their mean over the chip; a whole number unless the chip lies where two satellite tracks overlap (about 1% of chips) |
| `n_clear` | of those, the images passing the cloud test (Cloud Score+ `cs_cdf ≥ 0.6`): exactly the images the median picture is made of | their mean over the chip |
| `n_clear_median` | | the median pixel's clear images: half the chip has at least this many. Less pulled by a few very well seen pixels than the mean |
| `empty_share` | no clear image: the median has nothing, the pixel is 0 in every band (black to the model) | the share of such pixels, 0 to 1 |
| `thin_share` | 1 or 2 clear images: the "median" is really one picture, or the average of two | the share of such pixels. `1 - empty_share - thin_share` is the share with 3 or more |
| `clear_day_mean` | the days of the clear images (0 = the period's first day) | their average over all clear pixel-images of the chip: the date the picture mostly shows. Missing if the chip has no clear image |
| `cs_mean` | Cloud Score+ `cs_cdf` of every image (1 = clear, 0 = cloud) | its average over all images and pixels: how clear the period was, cloudy images included. Continuous, unlike the 0.6 cut |
| `cs_clear_mean` | the same, over the clear images only | how clean the images in the median were: 0.95 is crisp, near 0.6 means haze or thin cloud just passed the test |
| `ndvi_mean` | (B8 − B4) / (B8 + B4) of the composite: greenness, −1 to 1 | average over the pixels with data. Forest ≈ 0.8, bare soil or water near 0 or below |
| `bsi_mean` | ((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2)): bare soil index | average over the pixels with data. Higher = more bare soil |
| `cirrus_mean` | B10 (cirrus band) reflectance of the composite, 0 to 1 | average over the pixels with data. Near 0 is clean; higher means high haze left in the picture |

The three last columns describe the picture itself, so they are controls and secondary
outcomes as much as quality measures. All of these come from the one Earth Engine
request per tile (two of them, `cs_mean` and `cs_clear_mean`, from two extra bands; the
indices from the pixels already downloaded).

Worked examples:

- `n_obs 9, n_clear 0.45, empty_share 0.55` (May 2026, test_region). 9 images covered
  the chip. 55% of its pixels had no clear image at all; the other 45% had one each:
  0.45 × 1 + 0.55 × 0 = 0.45. The model saw a chip that is more than half black.
- `n_obs 7, n_clear 1.29, empty_share 0.70`. 7 images covered the chip. 70% of its
  pixels were cloudy in all 7; the other 30% had on average 1.29 / 0.30 ≈ 4.3 clear
  images each. A chip half under a cloud that never moved, half clear most of the time.
- `n_obs 18, n_clear 12.0, empty_share 0, clear_day_mean 70` (a made-up quarter, Q1).
  Every pixel has data, from about 12 clear images per pixel. The clear images were
  taken on average on day 70, i.e. 12 March: the picture shows March more than
  January, because January was cloudy. If the same chip's Q2 picture has
  `clear_day_mean 20` (21 April), the two pictures are only 40 days apart, not 91, and a
  change between them happened in those 40 days.

So **`n_clear` is not a share**: 0.45 means "0.45 clear images per pixel on average",
not "45% clear". Three useful quantities follow from the columns, with no new download:

- `1 - empty_share`: the share of the chip the model actually saw.
- `n_clear / (1 - empty_share)`: the clear images per pixel *where there was any*, i.e.
  how solid the median is on the visible part.
- `period start + clear_day_mean days`: the date the composite mostly shows. Compare it
  across periods before reading a change as growth in that period.

Expect `n_clear <= n_obs`, `empty_share` near 0 in the dry season, `clear_day_mean`
between 0 and the period's length, and missing exactly where `n_clear` is 0. A
month's `n_obs` and `n_clear` add up exactly to its quarter's (measured, `docs/findings.md`
§1); the other columns do not (averages, and each period counts its days from its own start).
Runs scored before 23 Oct have only `n_obs`, `n_clear` and `empty_share`.

#### Reading it in Python (laptop or Goethe, the same code)

detect's output, every period of one frequency (only the columns you need, to save memory):

```python
import pandas as pd
run = "tests_output/test_region_2026-10-19"          # on Goethe: "/work/<group>/<user>/runs/<run>"
cols = ["chip_id", "confidence", "n_clear", "empty_share", "thin_share", "clear_day_mean", "cs_mean"]
d = pd.read_parquet(f"{run}/detect/quarterly", columns=cols)   # adds period and subregion
d.groupby("period")[["n_clear", "empty_share", "clear_day_mean"]].describe()
(d.confidence > 0.4).groupby(d.period).sum()          # AMW's raw detections per quarter

one = pd.read_parquet(f"{run}/detect/quarterly/period=2026Q1/subregion=2")   # one task
some = pd.read_parquet(f"{run}/detect/quarterly", columns=cols,
                       filters=[("period", "in", ["2026Q1", "2026Q2"])])     # a few periods
```

The assembled panel (after A7), layers side by side, with `scored` and AMW's flags:

```python
import sys; sys.path.insert(0, "panel_pipeline/lib")  # from the repository root
from read_panel import read_panel
q = read_panel(run, "quarterly")                      # scores + postprocess, every period
q = read_panel(run, "quarterly", periods=["2026Q1"],
               columns={"scores": ["confidence", "n_clear", "empty_share", "clear_day_mean"],
                        "postprocess": ["det_raw"]})
```

`panel_pipeline/checks/explore_runs.ipynb` does this with maps and a monthly ↔ quarterly check.
Reading a whole period of the full Amazon this way needs ~18 GB of memory; on a laptop,
read one `period=` folder, or a few columns. On Goethe, anything bigger than a test area
is read inside a job (B6), not on the login node.

### A7. Assembly (next step)

`assemble_period.py` (first version, tested on the two small test runs) takes one (frequency,
period), refuses unless every subregion's task is complete, and writes one file per
period with every planned chip: AMW's detection flags, the isolation distance, and
`scored = False` rows for failed tiles. Nothing is ever deleted (Rmd P1, P2, D11).
What is decided and what is open is in `docs/open_design_decisions.Rmd`.

---

## Part B: on the Goethe cluster

Goethe-NHR, Frankfurt: <https://wiki.csc.uni-frankfurt.de/wiki/doku.php?id=public:usage:goethe>.
Login `goethe.hhlr-gu.de`; Slurm; whole 40-core nodes (`general1`, 192 GB, 21 days
max); `test` partition for short tests (2 h, 2 jobs). Compute nodes reach Earth Engine.
The scripts for the cluster are in `cluster/`.

### B1. What is different

| | Laptop | Goethe |
|---|---|---|
| Where commands run | your terminal | inside a Slurm **job** on a compute node; the login node is only for `git`, installs, `status.py` and small checks |
| Python | `venv/` in the repo | a venv on `/work` (fast, big); the repo in `/home` (backed up) |
| Earth Engine login | `earthengine authenticate` (browser) | a **service account key** file, named by `GOOGLE_APPLICATION_CREDENTIALS` (see `notes/service_account_guide.md`) |
| `run.output_root` | `tests_output` | `/work/<group>/<user>/runs` |
| Cores | all of the laptop | a whole node, 40 cores, so **several tasks run side by side on one node** and TensorFlow must be told how many cores each may use (`cluster/goethe_env.sh`) |
| Stopping | Ctrl-C | `scancel <jobid>` sends the same stop signal: scored tiles are saved |
| Closing the laptop | stops the run | nothing: jobs run without you |
| Earth Engine load | one process | the **sum over all running tasks** of their `--workers` must stay under the project's limit (B4) |

### B2. One-time setup

The full click-by-click version is in the "Cluster migration" thread. In short:

1. Account and first login (`ssh <user>@goethe.hhlr-gu.de`; the first password must be
   changed at once). Note your group name: `quota` and `ls /work` show it.
2. SSH key on Goethe added to GitHub, then `git clone` the repository into `/home` and
   `git checkout pipeline-rebuild`.
3. uv and the venv on `/work`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export UV_CACHE_DIR=/work/<group>/<user>/cache/uv UV_PYTHON_INSTALL_DIR=/work/<group>/<user>/python
   uv python install 3.11
   uv venv --python 3.11 /work/<group>/<user>/venvs/amw-panel
   source /work/<group>/<user>/venvs/amw-panel/bin/activate
   uv pip sync requirements/panel-linux.txt
   ```
4. The service-account key in `~/.secrets/ee-amw-panel.json` (`chmod 600`), then
   `python cluster/check_login.py` on the login node, then once more inside a `test` job.
5. Fill in `cluster/goethe_env.sh` (group, user, key path) and `source` it in every
   shell and job.

### B3. A run on the cluster, step by step

```bash
source cluster/goethe_env.sh                                   # venv, key, thread limits
cp panel_pipeline/config.yaml configs/<run>.yaml               # then edit: name, output_root, clip_to
python panel_pipeline/lib/config.py --config configs/<run>.yaml

# plan: seconds for a test area; 2-3 h for all of Amazon_ACA, so inside a job, not on the login node
salloc --partition=general1 --nodes=1 --time=4:00:00 --mem=0   # interactive job (test: --partition=test --time=1:55:00)
srun python panel_pipeline/plan.py --config configs/<run>.yaml
exit

# the task list, period first
python panel_pipeline/status.py --run /work/<group>/<user>/runs/<run> --todo > /work/<group>/<user>/runs/<run>/tasks.txt

# first test: one task, interactively, watching the progress lines
salloc --partition=test --nodes=1 --time=1:55:00 --mem=0
srun python panel_pipeline/detect.py --run /work/.../<run> --frequency quarterly --subregion 6 --from 2024Q3 --to 2024Q3 --workers 20 --limit 200
exit

# then unattended: N tasks of tasks.txt at a time on one node, W workers each
sbatch cluster/detect_tasks.sbatch /work/.../<run>/tasks.txt 2 20
squeue --me                                                    # is it running?
python panel_pipeline/status.py --run /work/.../<run>          # how far? (login node, any time)
tail -3 /work/.../<run>/detect/quarterly/period=2024Q3/subregion=6/_log.txt
```

`cluster/detect_tasks.sbatch` runs the lines of a task file on **one node**, N at a time,
appending `--workers W` to each. A task that fails is simply still in the next
`status.py --todo`. **Restart = make a new task list and submit again**; finished tasks
are skipped in seconds. The tasks in flight × W is the Earth Engine load (B4).

### B4. The Earth Engine ceiling (read before planning a big run)

The cluster is not the bottleneck; Earth Engine is. Two documented limits apply to the
project, whoever and wherever the requests come from
(<https://developers.google.com/earth-engine/guides/usage>,
<https://developers.google.com/earth-engine/guides/noncommercial_tiers>):

- **40 concurrent requests per project** by default (an adjustable quota). More
  requests queue and answer "429 Too Many Requests"; detect.py counts those as
  *retries* in its progress line. Your Mac ran 22 workers with 0 retries (22 Oct).
- **100,000 EECU-hours per month** on the Partner tier. Not a hard cut-off: over it, the
  project runs in "restricted mode" (slower) until the 1st of the next month.

Measured on 22 Oct (`checks/bench_download.py`, subregion 3, 2026Q3): one tile-quarter costs
Earth Engine about **130 EECU-seconds** (56–213) and ~35 s per request on its own;
18.6 MB of pixels. A month is cheaper (fewer images in the median) but not measured yet:
`python panel_pipeline/checks/bench_download.py --period 2026-07 --tiles 5`.

So, per day, with 40 requests in flight at ~35–60 s each: **about 60,000–100,000
tile-periods**, and the monthly EECU quota allows about 90,000 tile-quarters a day.
Scoring that needs about 12 cores, so **one node keeps up**; more nodes only help if
Google raises the concurrency quota. The full plan (274,932 tiles × 140 periods = 38.5 M
tile-periods) is therefore about a year of Earth Engine time; quarterly only (9.6 M) about
3–4 months. This is the number to decide the scope with.

### B5. Storage and backup

| Path | Holds | Notes |
|---|---|---|
| `/home/<group>/<user>` | the repository, `~/.secrets/`, copies of each run's `run/`, `plan/tiles.csv`, `_status.json` and `_log.txt` files | 30 GB, backed up nightly, slow |
| `/work/<group>/<user>` | the venv, uv's cache, the run folders | 5 TB group quota, fast, **no backup**, old data expected to be deleted |
| `/arc0x/archive/<group>` | finished periods and the panels, copied with `rsync` from a login node | needs group space (group leader requests it) |

Size: one complete task is one `part_00000.parquet`; a tile-period is ~22 KB (529 rows ×
~42 B). Full Amazon detect ≈ 0.8 TB, 840 files. Nothing is deleted after assembly (Rmd
D11), so plan for about 2 TB on `/work` plus the archive copies.

### B6. Getting the results: on Goethe, or copied to a laptop

**On Goethe**, read with the code of A6, in a job: the login node is shared and kills
big processes.

```bash
salloc --partition=test --nodes=1 --time=1:55:00 --mem=0     # an interactive node
source cluster/goethe_env.sh
srun python -c "import pandas as pd; d = pd.read_parquet('/work/<group>/<user>/runs/<run>/detect/quarterly', columns=['period', 'n_clear']); print(d.groupby('period').n_clear.describe())"
exit
```

**To a laptop**, copy with `rsync` from the laptop's terminal (it copies only what is new
or changed, so run it again at any time). The detect and panel folders keep their
layout, so the same A6 code reads the copy:

```bash
# one frequency's detect output, or one period of it
rsync -av <user>@goethe.hhlr-gu.de:/work/<group>/<user>/runs/<run>/detect/quarterly/ \
          tests_output/<run>/detect/quarterly/
rsync -av <user>@goethe.hhlr-gu.de:/work/<group>/<user>/runs/<run>/detect/quarterly/period=2026Q1/ \
          tests_output/<run>/detect/quarterly/period=2026Q1/
# the assembled panel, plus the plan (chip squares, for maps)
rsync -av <user>@goethe.hhlr-gu.de:/work/<group>/<user>/runs/<run>/{panel,plan,run} tests_output/<run>/
```

Copy only finished tasks (`#` in status.py): a running task's part files change every
few minutes. A subregion-quarter of the full run is roughly 1 GB, all of quarterly
detect ~230 GB (`docs/findings.md` §7), so on a laptop copy a few periods or one subregion.

### B7. If something breaks

| Symptom | Cause | What to do |
|---|---|---|
| `FileNotFoundError: GOOGLE_APPLICATION_CREDENTIALS=…` | key path wrong | `ls -l ~/.secrets/`; fix `cluster/goethe_env.sh` |
| `403 … permission` at login | the service account lacks a role | `notes/service_account_guide.md` A3 |
| many *retries* in the progress line, tiles failing with 429 | over the concurrency limit | fewer tasks at once or fewer `--workers`; the sum must stay ≤ 40 |
| `slurmstepd: … exceeded memory limit` | job killed for memory | use `--mem=0` (whole node) or fewer tasks per node; one task needs ~2–4 GB |
| job ended at its time limit | `SIGTERM` | nothing lost: resubmit (`status.py --todo` → `sbatch`) |
| `S` in status.py | a task died without cleaning up (node lost, `kill -9`) | rerun its command; the lock is taken over after 15 min |
| `SKIPPED: … is being written by …` | two copies of the same task | let the first finish; this one did no harm |
| `PLAN ERROR: … already exists` | planning the same run twice | a run is planned once; choose a new `run.name` |
| `DETECT ERROR: … has changed since this run was planned` | the model file differs | one run uses one model; new run |
| `quota` shows `/work` nearly full | | archive finished periods, ask the group about the quota |
| `DETECT ERROR: … written by detect.py before the quality columns of 23 Oct` | the folder holds parts from the older detect.py | score into a new run, or move that folder aside |
