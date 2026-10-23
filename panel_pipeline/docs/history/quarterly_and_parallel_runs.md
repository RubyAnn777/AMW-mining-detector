> **History, not maintained.** 19 Oct. Naming and layout moved to `../../README.md`; decisions to `../STATE.md`; its still-open review items (A1, A4–A6, B2, B3, C1, C5, D4, D5) are copied into `../open_design_decisions.Rmd` (section 'Leftovers').

# Quarterly runs, folder naming, and what the parallel run will look like

Written 19 Oct 2026 for Ruby and the team. It is based on the code on `pipeline-rebuild`
(`panel_pipeline/`), on Ruby's local test run `test_region_small_19.10.2026/pilot` (4 tiles,
2,116 chips, subregion 2, monthly 2026-01 to 2026-03), and on the decisions in
`notes/handoff_manual.md`. Every claim about the code cites `file:line`; numbers marked
*measured* come from that test run, and numbers marked *estimate* are extrapolations.

Contents:
1. Run the quarterly scoring for the test region
2. Names: what each part of a path means, and a convention
3. The full picture once we parallelise
4. Scale: measured and estimated
5. Are we caching anything?
6. Review: inconsistencies, UX, edge cases, efficiency, git, memory
7. The notebook (`panel_pipeline/explore_runs.ipynb`)
8. Decisions for Ruby

---

## 1. Run the quarterly scoring for the test region

**No code changes are needed.** Your run already contains quarters: its frozen config says
`frequencies: [monthly, quarterly]` (`<run>/run/config.yaml`), so `detect.py` accepts
`--frequency quarterly` for it. From the repository root:

```bash
python panel_pipeline/detect.py --run test_region_small_19.10.2026/pilot \
    --frequency quarterly --subregion 2 --from 2026Q1 --to 2026Q1
```

What happens, step by step:

1. `argparse` reads the options (detect.py:304-312). `--frequency` must be `monthly` or
   `quarterly` (`choices=`), and anything else exits with code 2 before any work.
2. The run's own config copy is loaded (detect.py:318), never the live `config.yaml`.
3. `pick_periods` (detect.py:274) turns `2026Q1` into a `Period` and keeps every quarter
   from `--from` to `--to`, **both ends included**. A quarter must be written `2026Q1`;
   `--from 2026-01` with `--frequency quarterly` stops with "--from and --to must be quarterly
   periods" (detect.py:280).
4. The tiles for subregion 2 come from `plan/tiles.csv` (detect.py:290). They are the same 4
   tiles as the monthly run, so the chips and `chip_id`s are identical.
5. Earth Engine builds **one** composite for 1 Jan to 1 Apr 2026 (the end is excluded, so all
   of 31 March is in). It is AMW's median with the cloud test, plus our two count bands, from
   the same request (detect.py:52-67). A quarter's median is not the median of its three
   monthly medians, so this has to be its own request. That is why a quarter costs as many
   requests as a month.
6. The results go to a new folder, next to the monthly ones:

```
test_region_small_19.10.2026/pilot/detect/
├── monthly/
│   ├── period=2026-01/subregion=2/   part_00001.parquet  _status.json
│   ├── period=2026-02/subregion=2/   …
│   └── period=2026-03/subregion=2/   …
└── quarterly/
    └── period=2026Q1/subregion=2/    part_00001.parquet  _status.json   ← new
```

Expect about 20 seconds, like each month took (*measured*: 19-20 s per month for 4 tiles).
A quarter's composite uses about three times as many images, so Earth Engine may take a little
longer per tile.

**Check it** in the notebook (section 7): section 6 of the notebook joins every monthly row to
its quarter and tests an identity that must hold exactly. A quarter's window is its three
months' windows put end to end, so per chip

```
n_obs(2026Q1)   = n_obs(2026-01)   + n_obs(2026-02)   + n_obs(2026-03)     (7 + 6 + 7 = 20 here)
n_clear(2026Q1) = n_clear(2026-01) + n_clear(2026-02) + n_clear(2026-03)
```

If that fails, either the dates are wrong or Earth Engine's data changed between the scoring
runs. Both are worth knowing before scaling up.

### Why the quarterly scores go in the same run, not a new one

A **run** is one `plan.py` result: one chip universe, one model fingerprint, one frozen config.
Monthly and quarterly are two *outputs* of one run, kept in separate folders and later in two
panels (handoff §4, "Frequencies"), but sharing `plan/`. That guarantees they line up chip by chip.
A second run for the quarters would give the same `chip_id`s, because the grid is global and
fixed. It would still duplicate the plan, its provenance and its fingerprints, and leave two records that could
disagree.

---

## 2. Names: what each part of a path means, and a convention

```
<output_root> / <run.name> / detect / <frequency> / period=<tag> / subregion=<n> / part_00001.parquet
  where it is    which run              which panel   which period    which job        which batch
  stored         (identity)
```

- **`output_root`**: *where* runs are stored. It is a storage location that rarely changes:
  `tests_output` on your Mac, a scratch path on the cluster. Relative paths start at the
  repository root (config.py:250).
- **`run.name`**: *which* run. This is the identity. It may only contain letters, digits, `-`
  and `_` (config.py:145), because it becomes a folder name.
- Everything below the run folder is **computed, never typed** (handoff §6, "Routing
  principle"): from the frequency, period and subregion.

### What you have now, and why I'd change it from the next run on

Your test has `output_root: "test_region_small_19.10.2026"` and `run.name: "pilot"`. The two
roles are swapped: the descriptive name sits in the storage slot, and the identity is the
generic `pilot`. Three concrete consequences:

1. **Git sees it.** `.gitignore` ignores `tests_output/` but not this folder, so `git status`
   lists `test_region_small_19.10.2026/` as new, and provenance records it as an uncommitted
   file in every record (`run/provenance.json` and each `_status.json` say so). Every test
   would need its own `.gitignore` line, and one `git add .` would commit data.
2. **Every run is called "pilot".** Two test runs can't be told apart by name. The notebook
   labels runs by their path for that reason.
3. **`19.10.2026` doesn't sort.** Day-first dates sort by day, so 01.11 comes before 19.10.
   `run.name` refuses dots anyway.

**Proposed convention:**

| | `output_root` | `run.name` |
|---|---|---|
| local tests | `tests_output` (ignored by git) | `<area>_<YYYY-MM-DD>[_<note>]`, e.g. `test_region_small_2026-10-19` |
| cluster test | `/scratch/<you>/amw_panel` (absolute) | `madre_de_dios_2026-11-03` |
| the real run | the same scratch root | `aca_full_2026-11-10` |

ISO dates (`2026-10-19`) sort correctly as text, the same reason periods are tagged
`2026-01` and `2026Q1`.

**For today:** finish the quarterly in the existing folder, so the notebook compares months
and the quarter from one plan. From the next test on, use the convention. Re-planning this
area takes seconds and 16 Earth Engine requests, so nothing is lost by switching.

**One config per run** (proposal). Right now every test edits the tracked `config.yaml`, so
it is always "uncommitted". `plan.py` already takes `--config` (plan.py:158), so keep one
small YAML per run in a committed `configs/` folder
(`configs/test_region_small_2026-10-19.yaml`). Git then records exactly what each run was,
and `config.yaml` stays the template.

---

## 3. The full picture once we parallelise

### 3.1 The unit of work is a folder

Every `detect/<frequency>/period=<tag>/subregion=<n>/` folder is written by exactly one
process, and nothing else ever writes there. That makes the work **embarrassingly parallel**:
tasks don't talk to each other, share no files, and can run in any order on any machine. A task
is just one command:

```bash
python panel_pipeline/detect.py --run <run> --frequency <f> --subregion <n> --from <tag> --to <tag>
```

`--from X --to X` (one period) already works, so **no code change is needed to parallelise
by period**. The full Amazon run is

| | subregions | periods | tasks |
|---|---|---|---|
| monthly | 6 | 105 (2018-01 … 2026-09) | 630 |
| quarterly | 6 | 35 (2018Q1 … 2026Q3) | 210 |
| **total** | | | **840** |

### 3.2 What the run folder will look like

```
/scratch/<you>/amw_panel/aca_full_2026-11-10/          one run
├── run/      config.yaml  provenance.json               frozen once by plan.py
├── plan/     tiles.csv  chips.parquet                   ~276k tile rows, ~145 M chips (~1.4 GB)
├── detect/
│   ├── monthly/
│   │   ├── period=2018-01/
│   │   │   ├── subregion=1/  part_00001.parquet … part_00092.parquet  _status.json
│   │   │   ├── …
│   │   │   └── subregion=6/
│   │   ├── …
│   │   └── period=2026-09/
│   └── quarterly/
│       ├── period=2018Q1/subregion=1 … 6/
│       └── …  period=2026Q3/
├── logs/     (proposal) one log per task, e.g. monthly_2018-01_s1.log
├── assemble/ (next script) one file per frequency and period: all 6 subregions merged,
│             border duplicates removed, postprocess variables added
└── panel/    (later) the two final panels, split by area so each chip's series is together
```

The `key=value` folder names are **Hive partitioning**, a convention every Parquet reader
understands: pandas, pyarrow, DuckDB, R's `arrow`, Spark. Reading `detect/monthly` returns
`period` and `subregion` as columns without them being stored in every row, and a filter like
`period == "2024-07"` opens only that folder.

### 3.3 How the cluster would run it (sketch; the scheduler details are yours)

1. `plan.py` once (~2-3 h for all of Amazon_ACA, handoff §11).
2. A **task list**: one line per task, the 840 commands above. The notebook's section 4 builds
   it from the run's config already, so a 10-line `tasks.py` could write it to
   `<run>/tasks_monthly.txt`.
3. A **job array**: the scheduler starts task *i* by reading line *i* of that file. With
   Slurm, for example, `#SBATCH --array=1-630%40` runs 630 tasks with at most 40 at once, and
   each one runs `sed -n "${SLURM_ARRAY_TASK_ID}p" tasks_monthly.txt | bash`. The `%40` is the
   **throttle**, and it is our Earth Engine queue: the requests in flight are
   tasks at once × `max_workers`, so 40 × 8 = 320 here.
4. **Restart = resubmit.** Every task is **idempotent**: running it again skips saved tiles
   and retries failed ones (detect.py:242-253), and a finished task ends after a few seconds
   of checking. So "rerun the whole array" is always safe and redoes only what's missing.
5. **Progress** is read from the 840 `_status.json` files (the notebook's section 4, or a small
   `status.py` command; see §6 UX).
6. **Assemble** a period once all 6 of its subregion tasks are `complete`.

**Ordering trade-off.** You chose subregion first: all months of subregion 1, then 2, and so on
(handoff §4). Assemble needs all 6 subregions of a period, though, so with that order nothing
can be assembled or checked until subregion 6 starts. Ordering the task list period first
(2018-01 subregions 1-6, then 2018-02 …) lets assemble and the quality checks start after the
first few hours, so a problem shows up on day 1 rather than week 3. With one task per period,
the order is just the order of lines in the task file, so this costs nothing to change.

### 3.4 The one collision rule

Two processes must never write the same folder at the same time. Today nothing *enforces*
this. `saved_tiles` (detect.py:162) only notices duplicates the next time a process starts.
Worse, two processes can pick the same part number (detect.py:178-179) and one silently
overwrites the other's file. That is a **race condition**. Resubmitting the array while it is
still running would cause exactly that. The fix is a **lock file**; see §6, item A3.
**Update 19 Oct:** built (`lock.py`); see `notes/explainers/workers_locks_and_status.md`.

---

## 4. Scale: measured and estimated

| quantity | value | source |
|---|---|---|
| chips per tile | 529 (23 × 23) | *measured*, checked in the notebook |
| time per month, 4 tiles, `max_workers: 6`, MacBook | 19-20 s | *measured* (`_status.json` sessions) |
| disk per chip-period row (zstd Parquet) | ~42 bytes | *measured* on the test parts |
| memory per row in pandas | ~124 bytes | *measured* in the notebook |
| chip-period rows, Amazon_ACA | ~15.2 bn monthly + ~5.1 bn quarterly | 145 M chips × 105 / 35 periods |
| detect output on disk | ~0.6 TB monthly + ~0.2 TB quarterly | *estimate* from 42 B/row |
| one whole period in pandas | ~18 GB | *estimate*: so never load a full period at once |
| Earth Engine requests | ~276k listed tiles × 140 periods ≈ **38.7 M** | handoff §11 |
| part files | ~77,000 (840 folders × ~92 batches of 500 tiles) | *estimate* |
| CPU | ~2.3 s per tile on 4 cores → ~25,000 four-core-hours | handoff §6 *estimate*; not measured on the cluster |

The 20 s for 4 tiles is mostly waiting on one round of Earth Engine, not throughput: all 4
tiles downloaded at once. **The next measurement that matters** is a test with a few hundred
tiles (Madre de Dios, 748 tiles), which shows tiles per minute at a steady state. Try
`max_workers` at 4, 8 and 16 and watch for HTTP 429 ("too many requests") errors in
`_status.json`. That measurement decides how many tasks can run at once.

---

## 5. Are we caching anything?

**No.** Nothing is cached between runs:

- **Imagery: no cache.** `extractor_for` passes `image_cache_dir=None` (detect.py:83), so AMW's
  `get_tile_data` skips both its read and write branches (core/gee.py, `get_tile_data`:
  `if image_cache_dir:`). Each tile's pixels live only in memory, are scored, and are dropped.
  This was decided on purpose (handoff §4, "Image cache"). A cache would be ~200-300 TB, and
  AMW's stores rounded whole-number pixels, so cached scores would differ slightly from fresh
  ones.
- **What makes restarts cheap instead** is the part files: they work like a cache of
  *results*. A rerun reads which tiles are saved and skips them (detect.py:242-247), so only
  unfinished tiles are downloaded again.
- **Earth Engine** may cache on Google's side, but nothing relies on it.
- **The model** is loaded once per process (detect.py:343), so one task loads it once for all
  its periods.
- `__pycache__/` folders hold compiled Python code, not data, and are ignored by git.

What we give up: scoring the same imagery again (a new model, a bug fix) means downloading it
again. A small *permanent* sample cache, for example 100 tiles × all periods ≈ 70 GB, is still
an open option for model comparisons (handoff §4).

---

## 6. Review: inconsistencies, UX, edge cases, efficiency, git, memory

Priority: **High** = fix before the cluster run; **Med** = before the full run would be
nice; **Low** = polish. Nothing here blocks the quarterly test.

### A. Inconsistencies

| # | Finding | Proposal | Prio |
|---|---|---|---|
| A1 | `--run` is resolved from the folder you're *standing in* (detect.py:314, `Path(args.run).resolve()`), while `output_root` and every config path start at the repository root (config.py:250, :113). Run detect.py from another folder and the same `--run` points elsewhere. | Resolve a relative `--run` from `REPO`, like the config does. One line. | Med |
| A2 | Scientific and run-time settings are frozen together. `max_workers`, `batch_tiles` and `tries` don't change any score, but they sit in the frozen config, so a run planned on your Mac with `max_workers: 6` must be re-planned to use 16 on the cluster. You already hit this (commit `30fdb3c`, "changed workers"). | Let detect.py take `--workers` / `--batch-tiles` to override them, and record the values used in each session in `_status.json`. Settings that change a score (imagery, stride, thresholds, model) stay frozen. | High |
| A3 | Handoff §14 says "never run two detect.py copies on the same (frequency, subregion); the duplicate guard will stop the second". The real unit is the (frequency, period, subregion) folder, and nothing stops a second copy. It is only detected afterwards, and the part-number race (§3.4) can lose a file silently. | A lock file per folder. Create `_lock` atomically (`os.open(..., O_CREAT \| O_EXCL)` fails if it exists) holding host, process id and job id; delete it at the end. A lock left by a crash is recognised because its process or job no longer runs. | High |
| A4 | The monthly folder is `detect/monthly`, not `detect/frequency=monthly`, so it is the one level that isn't self-describing. Reading `detect/` as a whole would mix both frequencies with no column to tell them apart. | Either keep it and always read one frequency at a time (the notebook does), or rename it to `frequency=monthly` so it becomes a column like the others. I'd keep it: the two panels are never mixed by design. | Low |
| A5 | Partition columns come back as text categories by default (`subregion` is a *text* `"2"`), so monthly and quarterly tables concatenate into untyped columns. | Read with an explicit partition schema (`period` text, `subregion` integer). The notebook does; assemble should copy it. | Med |
| A6 | Moving a run folder leaves its frozen config naming the old `output_root` and `run.name`; detect.py doesn't compare them. | Harmless now. Either check in detect.py that the folder name equals `run.name`, or drop `output_root` from the frozen copy's meaning. | Low |

### B. UX

| # | Finding | Proposal | Prio |
|---|---|---|---|
| B1 | There is no quick way to see progress over 840 tasks without opening JSON files. | `status.py --run <run>`: one line per frequency, then a period × subregion grid (`✓`, `·` not started, `!` failures, `%` partial). The notebook's section 4 does this already and can be lifted into a script. | High |
| B2 | No dry run: to see what a command *would* do you have to log in to Earth Engine and start scoring. | `--dry-run`: print the periods, tiles, folders and their current state, then stop before the Earth Engine login. | Med |
| B3 | `--from 2026-01` with `--frequency quarterly` is refused, which is correct but strict. | Keep refusing, but name the right tag in the message: "did you mean 2026Q1?". `Period.quarter()` already exists. | Low |
| B4 | Logs go to the terminal only. On the cluster they go wherever the scheduler puts them. | One log per task inside the run (`<run>/logs/monthly_2018-01_s1.log`), so a run folder is self-contained. With Slurm that's just `--output=`. | Med |
| B5 | Output is only "complete or not". An empty quarter in the rainy season looks complete. | Print the period's mean `n_clear` and `empty_share` in the closing line of each period, so a cloud-wrecked period is visible in the log. | Low |

### C. Edge cases

| # | Case | What happens now | Prio |
|---|---|---|---|
| C1 | **Recent periods.** Sentinel-2 scenes and Cloud Score+ images arrive in Earth Engine with a delay, and are sometimes reprocessed. 2026Q3 scored today may differ from 2026Q3 scored in a month. | Each session records when it ran (`_status.json` → `sessions`). Rule of thumb: score a period only once it is a few weeks old, and re-score the last quarter before the final panel. The notebook's run comparison measures any drift. | Med |
| C2 | **Chips with no clear image** (`empty_share` near 1, `n_clear` near 0). The model scores black pixels and returns a number. | It is written, as it should be ("no data" ≠ "no mine"). Analysis must use `n_clear`/`empty_share`, e.g. treat the score as missing below one clear image on average. The notebook reports this share (`low_obs`). | Med |
| C3 | **Border tiles.** A tile on a subregion border is scored by two tasks, under both subregions. | Expected (AMW does the same). Assemble removes the duplicate. The notebook's checks are per subregion, so this doesn't trip them. In your test area there are none. | — |
| C4 | **Killed mid-batch** (time limit, out of memory). | Up to one batch is lost (500 tiles) and redone on restart. A half-written file starts with `_` and is deleted at the next start (detect.py:218-219). At ~2 s per tile, that's up to ~15-20 min of work per kill. A smaller `batch_tiles` on the cluster (A2) trades more files for less loss. | Med |
| C5 | **A task longer than the cluster's time limit.** One subregion-period is ~46k tiles; at ~2.3 s per tile that is ~29 h, above many clusters' 24 h limit. | Resubmitting resumes, so it still finishes, in two pieces. If it's a nuisance, the next step is splitting a period into tile shards (`--shard 3/8`), with part files named per shard so shards can't collide. | Med |
| C6 | **`--limit` test, then a full run.** | Works: the limit run leaves the folder `complete: false`, and the full run continues from it. The notebook shows it as "partial". | — |
| C7 | **Partial last quarter.** `last_month: 2026-08` with quarterly would leave Q3 incomplete. | Refused by `period_range` (periods.py, "does not end a quarter"). | — |
| C8 | **Two runs with overlapping areas** (e.g. the test region inside the full run). | `chip_id` is global, so they join. The notebook flags (chip, period) pairs scored by both and shows how far apart they are. Keep both rows with a `run` column rather than silently picking one. | Low |
| C9 | **A boundary file or the model changes between plan and detect.** | The model is refused by its sha256 (detect.py:265). Boundary files only matter at plan time and are fingerprinted then. | — |

### D. Efficiency and strategy

1. **Parallelise by period, not by subregion loop** (High; your decision, see §8). The handoff
   plan is one process per (frequency, subregion) looping all periods. For the full region
   that's ~46k tiles × 105 months per process: months of wall time for each of 12 processes.
   One task per period (840 tasks) needs no code change and lets the cluster use as many
   machines as it gives you.
2. **Earth Engine is the likely bottleneck, not CPU.** 38.7 M requests is the hard number.
   Measure tiles per minute against `max_workers` (§4) before picking the throttle.
3. **Order the task list period first** (§3.3) so assemble and the checks start early.
4. **Quarterly is the cheap safety net.** 35 quarters cost a third of 105 months in requests
   and suffer less from clouds. If time or budget gets tight, a complete quarterly panel is
   worth more than a monthly one with holes. That's an argument for running quarterly first,
   or interleaving it, rather than strictly after all months. Your call: it reverses the order
   decided in handoff §4.
5. **GPU for inference** if the cluster has them. TensorFlow runs the same model on a GPU with
   no code change, and CPU time (~25k four-core-hours) would drop a lot. Detection doesn't
   *need* a GPU, but it's worth one test job.
6. **Fewer, bigger files at assemble.** ~77k part files are fine on scratch, but assemble should
   write one file per (frequency, period) and delete the parts once its checks pass (handoff
   §11). Shared file systems slow down with millions of small files, and some cap the count.
7. **Don't shrink the votes yet.** Storing votes as float16 would save ~25% of disk, but
   `confidence` would then not be exactly their mean. 0.8 TB is manageable; revisit only if
   storage is short.

### E. Git

| # | Finding | Proposal |
|---|---|---|
| E1 | Run outputs live inside the repo and outside `tests_output/` (`test_region_small_19.10.2026/`). | Keep all local runs under `tests_output/` (already ignored). Never commit Parquet: git keeps every version forever, so 100 KB of test data today is 100 KB in every clone always. |
| E2 | Your local fix of the `ee_project` line in `detect.py` is not committed, so every session records `panel_pipeline/detect.py` as uncommitted and the commit hash doesn't describe the code that ran. | Commit it before the next run. The whole point of `git_commit` in the records is that it names the exact code. |
| E3 | The live `config.yaml` is edited per test and always uncommitted. | One committed config per run in `configs/` (§2). |
| E4 | Notebooks store their outputs (tables, plots, and data) inside the `.ipynb` file. | Commit `explore_runs.ipynb` with outputs cleared (Kernel → Restart & Clear All Outputs, then save). For colleagues, export a filled-in copy: `jupyter nbconvert --to html --execute panel_pipeline/explore_runs.ipynb` and share the HTML, not the notebook. |

### F. Memory

- **detect.py is safe.** At most 2 × `max_workers` tiles are in memory at once (~20-40 MB each),
  plus the model and one batch of results (~20 MB for 500 tiles). That's well under 2 GB. One
  thing grows with the folder: `saved_tiles` reads every saved `chip_id` at the start and end
  of a period (~24 M IDs ≈ 200 MB for a full subregion). That's fine now, but reading the
  saved tile list from `_status.json` would avoid it.
- **One full period is ~18 GB in pandas** (§4). So **assemble must not load a period whole.**
  It should go subregion by subregion, with only the border tiles handled together, or use
  DuckDB/Arrow, which stream from disk. That's the main design constraint for the next script.
- **The final panel (~20 bn rows) never fits in memory**, and Stata can't hold it either.
  Analyses read filtered columns (DuckDB, Arrow, `arrow` in R), or work on aggregates or a
  sample. That's why the final panel should be split by area: one chip's whole time series
  then sits in one file.
- **The notebook** counts rows before loading and refuses anything over `MAX_ROWS` (20 M rows
  ≈ 2.5 GB), so opening it on a large run can't freeze a laptop.

---

## 7. The notebook: `panel_pipeline/explore_runs.ipynb`

To show colleagues the output. It only reads, so it's safe to open while detect.py is still
writing. Set `RUNS` in the first code cell (one or more run folders) and run all cells.

| Section | Shows | Checks |
|---|---|---|
| 1 Folder | every folder with file counts and sizes | — |
| 2 Run record | frozen settings, provenance, counts | flags uncommitted files at plan time |
| 3 Grid & IDs | `chips.parquet`, `chip_id` taken apart | unique, sorted, zone/column/row = tile key, chip = chip_index, 529 per tile |
| 4 Task list | the run's full task list (840 for Amazon), the state of each task, the next command | complete / partial / failures / not started |
| 5 Scored chips | data dictionary, per-period statistics, `describe()` | all planned chips present, none extra, no duplicates, confidence = mean of votes, 0 ≤ confidence ≤ 1, n_clear ≤ n_obs, no missing values |
| 6 Monthly + quarterly | long table (stacked) and wide table (each month with its quarter's columns) | sum of 3 months' `n_obs`/`n_clear` = quarter; detected in quarter vs in any month |
| 7 Several runs | settings and model side by side | same model and scoring settings, overlap and score drift |
| 8 Plots | map of chips coloured by score (month next to its quarter), detection share and clear images over time | — |
| 9 Scale | bytes per row → full-run disk and memory | — |

It was tested on a copy of your monthly output. The quarterly parts were made up from your
months for that test only, so the joins and the count check could run; your real quarter will
replace that. Everything passes on your data, and it also runs, skipping the quarterly
sections, when only monthly exists. It needs pandas, pyarrow, numpy, matplotlib and PyYAML
(all in your venv) plus Jupyter itself.

---

## 8. Decisions (Ruby, 19 Oct)

1. **Task unit:** one task per (frequency, period, subregion). No code change was needed.
2. **Task order:** period first (all subregions of a period, then the next period), monthly
   before quarterly.
3. **Naming:** `tests_output/<area>_<YYYY-MM-DD>`, one committed config per run in `configs/`.
4. **Built on 19 Oct** (branch `claude/project-thread-jwtt0k`): the run-time overrides (A2),
   the lock file (A3) and `status.py` (B1, including `--todo` for the task list). How they work:
   `notes/explainers/workers_locks_and_status.md`.
5. **Cluster setup:** Ruby will set it up differently from the sketch in §3.3 when moving there.
