> **History, not maintained.** the decisions and current state are in `../STATE.md`; open questions in `../open_design_decisions.Rmd`; how to run in `../../README.md`. Last updated 19 Oct; §3, §4, §7, §8, §9, §11–§14 are partly out of date (see STATE.md).

# Handoff manual: the econometric panel pipeline

**What this is:** every technical detail and decision from the thread "Which pipeline
stages we need" (8–19 Oct 2026), distilled so a new thread can continue exactly where
it stopped. It is a summary, not the source of truth for code. The tested code lives
in `notes/drafts/`, and the long-form reasoning in the notes listed in §15.

**State as of 2026-10-19 06:45 UTC.** The source thread had just answered Ruby's last
questions (the first Earth Engine test and how a script runs; §13). Ruby hasn't reported
results yet, so check that thread for anything newer before continuing.

---

## 0. How to use this manual

- Read §1–§3 first: the goal, how Ruby wants to work, and where the code is. §7a lists
  exactly which AMW files we import and must keep.
- §4 is the decision log. Don't re-open a decision listed there unless Ruby does.
- §12 lists open questions, and §13 says what comes next.
- When this manual and an older note disagree, this manual wins: it records later
  corrections. Superseded ideas are flagged in §14.

---

## 1. Goal and scope

- **Goal:** a clean econometric **panel** of AMW's (Amazon Mining Watch) mining-detector
  scores. One row is one chip in one period. The panel has quality/control variables and
  a robust, restartable inference system. Nothing gets published to a website.
- **Base:** AMW's own inference engine (`core/`), reused as much as possible.
  **Do not use the `timeseries/` folder**: Ruby considers it inefficient and inconsistent.
  Its main flaw is two Earth Engine calls per tile, one for the image and one for the clear
  count.
- **Periods:** monthly and quarterly, Jan 2018 to Sep 2026 (2026 ends at Q3). That's
  105 months and 35 quarters, 140 periods. No annual periods.
- **Region:** AMW's **Amazon_ACA** (6 subregions), including the Andes area with AMW's logic.
- **Compute:** a research compute cluster, not a VM. It is one cluster with several machines.
  Ruby will work out the cluster details herself, so don't ask about them.
- **Earth Engine:** partner tier, Google Cloud project `deforestation-and-mining`. Limits are
  not a big worry, but a request queue/throttle is still wanted later.
- **Scope now:** the pipeline up to and including the **persistence variables**.
  **SAM2 is deferred.** It will come later as a separate script that adds its own columns (§9).

---

## 2. How Ruby wants to work (follow these)

1. **Ruby types the main code herself.** Claude gives the code block by block or function by
   function, explains it line by line, and tests it. Claude does not create pipeline files in
   her branch. Granularity: small enough not to overwhelm, broad enough to keep context.
2. **Exception, the "small-level stuff":** Claude makes small fixes and edits to existing files
   (comments, config tweaks, helper moves) itself, and pushes them to Claude's branch for Ruby
   to pull (§3).
3. **Explain the computer science concepts** behind every step: classes, CLI parsing, threads,
   and so on. Ruby has no CS background and wants to "think like a senior dev". Give rationale,
   and strong objections for and against.
4. **Don't assume; ask** when something about the project is unknown.
5. **Read AMW's code, not its comments.** Ruby says the comments are old and unreliable.
   Cite `file:line`.
6. **Stay as close to AMW as possible** while reaching "academic econometric perfection".
   Mark every reuse and deviation **in the code comments** with these four markers:
   - `# AMW: file.py:line`: reused or copied from AMW
   - `# Same as AMW: ...`: kept on purpose, even where it looks odd
   - `# DEVIATION from AMW: what. Why.`
   - `# ADDITION: ...`: new, nothing dropped

   `grep -rn "DEVIATION\|ADDITION" panel_pipeline/` then produces the methods-section list.
7. **MVP first, run by hand.** No Slurm yet. Ruby starts each stage manually. Slurm and a
   `submit.py` come after the MVP works.
8. **Monthly and quarterly never mix** (§4). Run order: all months for subregion 1 over time,
   then subregion 2, and so on; quarterly only after that.
9. **Small explanations go in chat.** A longer explainer can be a Markdown file in `notes/`.
   Tested drafts go in `notes/drafts/` so nothing gets lost.
10. **Verify Ruby's pushes.** When she pushes, check her files against the drafts and say what
    differs.
11. Code style she asked for: built-in `list[...]`/`dict[...]` type hints, not `typing.List`.
    Use `from __future__ import annotations` as the first code line.
12. **Settings go in config.yaml as their own sections.** Ruby objected to tucking the Earth
    Engine project under `imagery:` and to environment variables.

---

## 3. Repository, branches, git workflow

- Repo: `RubyAnn777/AMW-mining-detector`.
- `main` has only the underlying AMW scripts.
- `code-walkthrough` is the only branch with AMW's stage driver (`scripts/pipeline.py`,
  `core/pipeline_config.py`, `core/periods.py`, `core/persistence.py`,
  `scripts/stage_outputs.py`). It was reviewed at commit `36718a8`.
- **`pipeline-rebuild`** is Ruby's working branch. It contains `code-walkthrough` and `main`.
  New code lives in **`panel_pipeline/`**. As of 19 Oct, its head is `10c7cc1` with
  `config.py`, `config.yaml`, `grid.py`, `periods.py`, `plan.py`, `questions.md` (Ruby's own
  notes on overlap and SAM2) and a committed `tests_output/` folder.
- **Claude's branch** for small edits is `claude/project-thread-ls8p9r` (based on
  pipeline-rebuild). Ruby pulls it with
  `git checkout pipeline-rebuild && git pull origin claude/project-thread-ls8p9r && git push`.
  Claude's commits on it:
  - `c91c77b`: clip_to, run-name check, deviation comments (already in pipeline-rebuild)
  - `ddd7f5d`: refuse repeated YAML keys; relative `output_root` resolved from the repo root;
    `tests_output/` in .gitignore
  - `c675f5e`: Earth Engine project setting; `provenance.py` split out of plan.py
  - `a59a9b6`: `earth_engine:` gets its own config section

  **As of 19 Oct, pipeline-rebuild does not yet contain `ddd7f5d`, `c675f5e` or `a59a9b6`**,
  and has no `detect.py` or `provenance.py`. Ruby still has to pull them and type detect.py.
  Claude offered to push straight to pipeline-rebuild instead; Ruby hasn't answered.
- Boundary files were missing on pipeline-rebuild (removed in archive commit `e15dae2`).
  Restore them with
  `git checkout bac9d30 -- data/boundaries/Amazon_ACA.geojson data/boundaries/Amazon_ACA data/boundaries/andes_supplemental.geojson`.
  Ruby did this.
- Ruby committed the test run (`tests_output/`). Claude recommended
  `git rm -r --cached tests_output` and left it as Ruby's call.
- Ruby's machine: a MacBook (Apple silicon), venv at `.../RA_Raji/AMW-mining-detector/venv`,
  Python 3.11, with package versions matching AMW's `requirements-lock_2026-09-18.txt`.
  Claude's container has **no Earth Engine login**: nothing has been billed from it.

---

## 4. Decision log

| Topic | Decision | Source / note |
|---|---|---|
| AMW stages used | Inference engine + their concat/dedupe + postprocess and persistence **as variables**. SAM2 later. Drop cog, persist-masks, stage, manifest, publish | Ruby, 8 Oct |
| Panel unit | AMW's **overlapping** 48 px chips with a 24 px stride (529 chips per 552+2×12 px tile). Reversible: every other chip *counted globally* (tile index × 23 + chip index) gives an exact 480 m non-overlapping grid | Ruby, 8 Oct. Objections in round2 §2: cluster SEs (tile or Conley), effective N about ¼, mechanical spatial correlation, isolation rule measures blob size, summed areas count pixels up to 4× |
| Every chip written | Write **every** chip's score, not just positives (AMW drops below-threshold chips at `inference_engine.py:803`) | So "no mine" ≠ "no data" |
| Ensemble votes | Keep all **6 member votes** per chip as float32 columns `vote_1..6`; `confidence` = their plain mean, float64 | Ruby, 8 Oct. Members are recorded in `_status.json` |
| Dedupe | Keep AMW's concat + dedupe (higher confidence wins, with its votes), but match on exact **chip_id**, not rounded centroids. Record the number of rows dropped | Ruby, 8 Oct |
| Andes | **No second inference run.** Same scores, threshold `> 0.2` on `in_andes` chips. Downstream also copy AMW: no isolation rule for Andes, separate SAM2 run later | Verified: threshold is the only inference difference (`pipeline.py:151-172`); the Andes area is 100% inside ACA; all 390 Andes tiles are in the subregion lists |
| Periods | Months + quarters, 2018-01 to 2026-09. **Separate Earth Engine composites** for each month and each quarter (different medians) | Ruby |
| Period end date | End = **1st of the next period**, which Earth Engine's `filterDate` excludes, so every day is imaged. Ruby: "not a bug", don't call it one | Matches AMW's earlier runs and training imagery (`..._2024-01-01_2025-01-01`, docs/PROVENANCE.md:402-411). Deviates from AMW's *current* driver, which passes the last day (pipeline.py:80-81, :160 via core/periods.py:84) and so drops it |
| Cloud recipe | Same for monthly and quarterly (AMW's median, Cloud Score+ `cs_cdf ≥ 0.6`), with clear counts as quality controls | Ruby |
| Quality vars | `n_obs`, `n_clear`, `empty_share` per chip, from **extra bands in the same Earth Engine request** (no second call) | Claude's default (15 Oct). Ruby didn't object; treat as accepted |
| Image cache | **None.** It would be about 200–300 TB, and AMW's cache stores uint16-rounded pixels. Resume from part files instead | Claude's default. Cost: rescoring later means a full re-download. Possible later: a small permanent sample (e.g. 100 tiles × all periods ≈ 70 GB) |
| Frequencies | Never mixed: separate folders and **two panels** (chip × month, chip × quarter) with the same chip_id, joined only at the end through a `quarter` column in the monthly panel | Ruby, 12 Oct |
| Run order | All monthly periods for subregion 1, then 2, …, 6; then quarterly | Ruby. Trade-off: assembling a month needs all 6 subregions, so first test a small area through every step |
| Work unit | AMW's: one process per (frequency, subregion), looping periods, 500-tile batches, each batch written as its own part file | Ruby: stay close to AMW. Slurm shards dropped for the MVP |
| Region | Amazon_ACA (8.24 M km²), not `amazon_basin` (6.70 M km², 95.4% of the Andes) | Ruby, 8 Oct |
| Test area | `region.clip_to: data/boundaries/la_pampa_corridor.geojson` (AMW's pilot area, 80 tiles, 42,320 chips, all in subregion 6). Larger test: `madre_de_dios_mining_aoi.geojson` (748 tiles, 395,692 chips, 32,101 Andes chips) | `null` = whole region |
| Config | One `config.yaml`, validated by `config.py`; **copied once into the run folder by plan.py**; later scripts read only `<run>/run/config.yaml` | So editing config.yaml can't change a half-done run |
| Earth Engine project | Own section `earth_engine: project: "deforestation-and-mining"`, read as `cfg.ee_project`; no default (AMW's gee.py:34 would bill upstream's `earthindex`) | Ruby, 15 Oct. Use the project *ID* if the console shows a different one |
| Persistence | k = 2 within `window_months` = 12, both configurable. **The exact rule is still open** (§12) | Ruby: "flexible" |
| postprocess | Kept as **variables** (loose 0.43/0.75, stringent 0.55/0.8, 5th neighbour beyond 3 km), not as a filter | Ruby |
| SAM2 | Deferred. Later, each period masks only its own detections; earlier periods are "backfilled" in the panel (Claude's reading; Ruby didn't confirm) | §9 |
| Slurm | Later. MVP is run by hand | Ruby, 8 Oct |
| `t_prov` | Left out (AMW's website "too recent to confirm" threshold); the panel just has missing forward values at the end | Claude, 12 Oct |

---

## 5. AMW's 13 stages and what we do with them

AMW's driver (`scripts/pipeline.py`, code-walkthrough only):
- "HUMAN" stages only **print** commands, which a person pastes into tmux on 2 VMs.
- "RUNNERS" run via `subprocess.run`.
- Stages find each other's files by **rebuilding filenames** from shared settings: "the filename
  is the contract". It is fragile, because a mismatch just means "no file found", with exit code 0.
- `--periods` (pipeline.py:562, argparse) selects from `ALL_CURRENT_PERIODS`
  (pipeline_config.py:53). `WHOLE_HISTORY` stages ignore it.

| # | AMW stage | Ours |
|---|---|---|
| 0 | review-config (prints settings) | `config.py` validation + `plan.py` frozen copy + provenance |
| 1 | inference | `detect.py` (AMW engine, adapted) |
| 2 | concat (stack 6 subregions + dedupe border tiles) | kept, inside `assemble_period.py` |
| 3 | filter (clip Andes 0.2 detections) | replaced by the `in_andes` flag + threshold |
| 4 | postprocess (dual threshold + isolation) | kept as variables, in `assemble_period.py` |
| 5 | persist-detections | kept as variables, in `persist.py` (rule redesigned) |
| 6–7 | mask-annual / mask-quarterly (SAM2) | later, one rule for all periods |
| 8–12 | cog, persist-masks, stage, manifest, publish | dropped. Manifest is replaced by the automatic run record |

What concat really fixes: it is **not** the chip overlap. AMW runs 6 subregion jobs, and a tile
straddling a border is created by both jobs. The subregions have zero area overlap (checked).
Border tiles in ACA: 1,186 (274,932 unique tiles, 276,120 listed).

---

## 6. Architecture (scripts and order)

```
config.yaml ─► plan.py            once per run: run folder, config copy, provenance, tiles, chips
               detect.py          per (frequency, subregion), loops periods; the only Earth Engine step
               assemble_period.py per period, needs all 6 subregions: completeness check →
                                  concat + dedupe on chip_id → postprocess variables
               persist.py         after all periods: persistence variables, final panel(s)
  later:       sam2_*.py          reads the run, writes its own (chip_id, period) table
  later:       submit.py / Slurm  job arrays with a %N throttle (= the Earth Engine queue),
                                  afterok dependencies
```

- **postprocess works across space** (3 km neighbours cross subregion borders), and **persist
  works across time** (all periods per chip). So they are separate steps.
- **Routing principle:** no output path is ever typed. Every path is computed from (run,
  frequency, period, subregion). Each job writes only into its own folder, stamps its identity
  into its records, and writes temp files then renames them (atomic). Assemble checks expected
  against present **by name** before merging.
- **Earth Engine load** = processes × download threads (`max_workers` = 8).
- The detect CPU estimate is about 2.3 s per tile for the model on 4 CPU cores. No GPU is needed
  for detection; GPUs only matter for SAM2.

Planned file list (from round2 §11; `tiles.py` became `grid.py`, `fetch.py` was folded into
`detect.py`): config/periods → grid/plan → detect → assemble_period (concat + postprocess) →
persist → panel → later sam2 + jobs/.

---

## 7. Module reference (status as of 19 Oct)

| File | Status | Content |
|---|---|---|
| `periods.py` | on pipeline-rebuild ✓ | `Period(year, first_month, n_months)`, frozen and ordered dataclass. Properties `frequency`, `tag` (`2018-01` / `2018Q1`, so text sorting = time order), `month_index` (year×12+month−1, one clock for both frequencies), `start`, `end` (1st of next period), `date_span`. Methods `months()`, `quarter()`; classmethod `parse(tag)`; `period_range(first, last, frequency)`, which refuses partial quarters. No third-party imports |
| `config.py` | on pipeline-rebuild, but missing Claude's latest 3 commits | Frozen dataclasses per YAML section: `Region` (subregion_files dict, andes_boundary, clip_to), `Imagery`, `Inference` (model_path, stride_ratio, tries, batch_tiles), `DualThreshold`, `Thresholds`, `Persistence`, `Config` (source, run_name, output_root, ee_project, periods dict, …). Helpers `_section` (exact keys: catches typos), `_int` (rejects bools), `_share` (0<x<1), `_file` (must exist, relative to REPO), and `_UniqueKeyLoader` (a key written twice is an error). Then `load_config()`, `describe()`, and the `__main__` check. The run name must match `[A-Za-z0-9_-]+`. `output_root` `CHANGE_ME` is refused, and a relative one is resolved from the repo root. Only `S2L1C` is allowed. `t_iso ≥ t_main`. `window_months` must be a multiple of 3 if quarterly, and k must be reachable in the window |
| `config.yaml` | on branch | Draft in `notes/drafts/config.yaml` (§8) |
| `grid.py` | on pipeline-rebuild ✓ | The space axis. Appends `core/` to `sys.path` (APPEND, because AMW's `core/periods.py` would shadow ours). Imports AMW's `create_tiles`, `cut_chips`, `centroid_key`. `CHIP_PX=48`, `RESOLUTION_M=10`. Functions `read_shape` (`union_all`), `tiles_for`, `chip_ids`, `split_chip_id`, and `chip_table` (a blank 576×576×1 image through AMW's `cut_chips` gives AMW's exact squares) |
| `plan.py` | on pipeline-rebuild (old version; the new one imports `provenance.py`) | `plan_tiles` (long tiles table; clip_to intersects each subregion; Andes tiles must be ⊂ subregion tiles), `write_chips` (streamed in 500-tile batches to Parquet; `shapely.prepare` is about 500× faster; checks that chip_ids increase strictly), `plan()` (builds `<name>.planning/`, renames when complete, refuses an existing run). Takes the CLI option `--config` |
| `provenance.py` | Claude's branch only | `environment(cfg)` (host, python, git commit, uncommitted files except the config, package versions), `fingerprints(cfg)` (sha256 of the model + boundaries), `sha256`, `repo_path` |
| `detect.py` | draft only (`notes/drafts/detect.py` + `detect_walkthrough.md`); **Ruby still has to type it** | Covered below |

**detect.py in short:**
- CLI: `--run` (required), `--frequency` (required), `--subregion` (required), `--from`,
  `--to`, `--limit`.
- `PanelExtractor` subclasses AMW's `gee.GEE_Data_Extractor` and overrides only
  `_build_composite`. It adds `n_obs` (count of `cs_cdf`) and `n_clear` (count with
  `cs_cdf ≥ 0.6`) bands from the same collections (`COPERNICUS/S2_HARMONIZED` +
  `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`). They arrive in the same `computePixels`
  request, with bands = AMW's 13 + 2 counts.
- `score_tile`: AMW's 4 lines copied (÷10000, clip 0..1, float32; `cut_chips` 48/24;
  `model.predict`; mean). Quality means come per chip via the same `cut_chips`.
- `score_batch`: a thread pool with at most 2×workers downloads in flight (about 0.3 GB,
  against AMW's about 9 GB per batch). The same downloads give the same scores.
- Part files are written temp-then-`os.replace`. `_status.json` records complete,
  tiles_planned/saved/failed (with error text), not_tried, and one entry per session
  (start/finish, environment).
- AMW's `tries`=3 passes; resume comes from the part files.
- `check_model` refuses if the model's sha256 differs from what plan recorded.
- The member order is read from the model's final Concatenate layer.
- It sets `gee.EE_PROJECT = cfg.ee_project` and logs in before making any folder, then calls
  `apply_darwin_tf_compat()` and loads the model with `compile=False`.
- Tested on fake imagery: bit-identical to AMW's `InferenceEngine.predict_on_tile_pixels` on
  3,174 chips; resume, retries, crashes and refusals all pass. **Real Earth Engine is untested.**

---

## 7a. AMW code we keep and import (checked against the import lines, 19 Oct)

Checked against `panel_pipeline/*.py` on `claude/project-thread-ls8p9r` (head `a59a9b6`) and
the `detect.py` draft. This is what is actually imported, not what the comments say.
**Don't delete, move or rename anything in the "imported" or "used as data" tables.**
AMW code is reached by `grid.py` appending `<repo>/core` to `sys.path`, and only `core/` is
on that path. So these must stay as plain files in `core/`, with these names.

### Imported directly by our code

| AMW file | What we import | Used in | How |
|---|---|---|---|
| `core/tile_utils.py` | `create_tiles` (line 102), `cut_chips` (line 213) | `grid.py` (tiles, chip squares); `detect.py` (cutting real pixels) | called unchanged |
| `core/postprocess.py` | `centroid_key` (line 113) | `grid.py` (`amw_key_x/y` columns) | called unchanged |
| `core/gee.py` | the module: `GEE_Data_Extractor` (line 179), `DataConfig` (line 105) and its `_BAND_IDS` (line 129), `EE_PROJECT` (line 34), `_ensure_earth_engine_initialized` (line 47) | `detect.py` | subclassed. `PanelExtractor` overrides only `_build_composite` (line 199) and calls the original through `super()`. `get_tile_data` (line 289) is inherited unchanged and makes the `computePixels` request (line 336) |
| `core/tf_darwin.py` | `apply_darwin_tf_compat` (line 19) | `detect.py` | called unchanged (macOS only) |

**Pulled in automatically by those imports** (also keep):
- `gee.py` imports `CenteredTile` and `ensure_tile_shape` from `tile_utils.py`.
- Nothing else in `core/` is loaded.

The third-party packages these AMW files need:
- `tile_utils.py`: `affine`, `descarteslabs`, `geopandas`, `matplotlib`, `shapely`
- `gee.py`: `earthengine-api` (`ee`), `google-api-core`, `rasterio`
- `postprocess.py`: **`scikit-learn`**, which is missing from AMW's requirements.txt

So importing `centroid_key` alone requires scikit-learn.

### Used as data, not code

- `models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`: the model. Its sha256 is pinned per run.
- `data/boundaries/Amazon_ACA/Amazon_ACA_{1..6}.geojson` (subregions) and
  `data/boundaries/andes_supplemental.geojson` (the `in_andes` flag).
- Test areas: `data/boundaries/la_pampa_corridor.geojson`, `madre_de_dios_mining_aoi.geojson`.
- `data/boundaries/Amazon_ACA.geojson`: not read by the config. Keep it for checking that the
  six subregions add up to the whole.

### Copied rather than imported (keep as the reference our comments cite)

- `core/inference_engine.py`: the 4 scoring lines (945-949, 766, 786: ÷10000 and clip, chips,
  predict, mean), stride (756), the model load (424), and the tries loop structure (1151-1157).
  **We don't import this file.** It imports `torch`, `sam2_logits`, `dense_embedding_cache` and
  others, so importing it would drag in the SAM2 stack. Our tests compared against its
  `InferenceEngine.predict_on_tile_pixels`.
- `core/periods.py`, `scripts/pipeline.py`, `core/pipeline_config.py`: they are referenced in
  deviation comments (dates, thresholds, region paths) but never imported.

### Planned for the next scripts (not imported yet)

- **`assemble_period.py`** will reuse from `core/postprocess.py`:
  - `dedupe_detections` (146), which concat uses (`scripts/concatenate.py:31`)
  - `centroids_xy_m` (190)
  - `kth_neighbor_km_on_catalog` (210)
  - `keep_dual_threshold` (231) / `dual_threshold_filter` (338)
  - possibly `PostprocessConfig` (38)

  Dedupe will be adapted to match on chip_id, so check whether it can take a key column or
  must be copied.
- **`persist.py`: don't import `core/persistence.py`.** At line 76 it does
  `from periods import QUARTER_SPANS, QUARTER_TAG_RE, …`. With our `panel_pipeline/` first on
  the path, that finds *our* periods.py and crashes. Line 79 also loads `pipeline_config`. Copy
  the rule's logic (lines 97-112) instead, with an `# AMW:` marker.
- **SAM2 later:** `core/sam2_mask.py`, `SAM2_Masker` and `MaskConfig` in
  `core/inference_engine.py` (lines 1289 and 247), `core/sam2_logits.py`, and the union rule in
  `core/sam2_build_cog.py`. That is when the torch/SAM2 dependencies come in.

### Not needed by the panel pipeline

- `timeseries/`. Ruby ruled it out.
- `scripts/pipeline.py` (the old driver), `scripts/stage_outputs.py`, `scripts/geo_filter.py`,
  `scripts/concatenate.py`. Their logic is replaced or reused through `postprocess.py`.
- `core/persistence_rule_*.py`, `core/sam2_persistence.py`, `core/dense_embedding_cache.py`,
  `core/model_library.py`, `core/inference_pipeline.py`, and the notebooks.

Leave these in the repo. Removing them buys nothing, and the comments cite some of them. This
"not needed" list comes from the import trace, not from a test that deleted them.

---

## 8. Config reference (current draft)

```yaml
run:          {name: "pilot", output_root: "tests_output"}     # Ruby's local test folder
earth_engine: {project: "deforestation-and-mining"}
periods:      {first_month: "2018-01", last_month: "2026-09", frequencies: [monthly, quarterly]}
region:
  subregion_files: "data/boundaries/Amazon_ACA/Amazon_ACA_{n}.geojson"   # pipeline.py:159
  subregions: [1, 2, 3, 4, 5, 6]
  andes_boundary: "data/boundaries/andes_supplemental.geojson"           # pipeline.py:165
  clip_to: "data/boundaries/la_pampa_corridor.geojson"                   # null = whole region
model:        {path: "models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5"}
imagery:      {collection: "S2L1C", tilesize: 552, pad: 12, clear_threshold: 0.6, max_workers: 8}   # gee.py:107-118
inference:    {stride_ratio: 2, tries: 3, batch_tiles: 500}   # inference_engine.py:181,185; pipeline.py:161
thresholds:
  raw: 0.4            # AMW cut is strictly ">" (inference_engine.py:803)
  andes: 0.2          # ">"
  loose:     {t_main: 0.43, t_iso: 0.75}   # postprocess uses ">=" (postprocess.py:250)
  stringent: {t_main: 0.55, t_iso: 0.8}
  neighbours_k: 5
  isolation_km: 3.0
persistence:  {k: 2, window_months: 12}
```

YAML traps:
- Quote dates and text: an unquoted `2018-01-01` becomes a date object, and `no`/`yes` become
  booleans.
- A key written twice is silently overwritten by PyYAML. Our loader refuses it.
- Note: "0.43" is the loose `t_main`, not a raw threshold. The raw cut is 0.4 everywhere
  (pipeline_config.py:73).

---

## 9. Data layout and schemas

```
<output_root>/<run.name>/
  run/config.yaml            frozen copy; the only config later scripts read
  run/provenance.json        planned_utc, environment, file_sha256, counts
  plan/tiles.csv             long: one row per (subregion, tile_key); border tiles listed twice
  plan/chips.parquet         chip_id, tile_key, chip_index (int16), minx/miny/maxx/maxy (full precision),
                             amw_key_x/y (AMW's centroid_key), in_region, in_andes
  detect/<frequency>/period=<tag>/subregion=<n>/part_00001.parquet …, _status.json
```

- **detect parts** have the columns `chip_id`, `confidence` (float64), `vote_1..6` (float32),
  `n_obs`, `n_clear`, `empty_share` (float32).
- The folders are Hive-partitioned, so `pd.read_parquet("<run>/detect/monthly")` reads all
  parts, adds `period` and `subregion` columns, and skips files starting with `_`.
- **chip_id** is an int64: `zone | tile column+500 | tile row+5000 | chip index`. Example:
  `194784737000` = zone 19, tile −22:−263, chip 0, the first La Pampa chip.
  - ACA range: zones 17–23, columns −61..60, rows −411..201.
  - The chip index follows AMW's `cut_chips` order, **column-major** (chip 1 is below chip 0,
    and chip 23 starts column 2).
  - It is an integer because text IDs would cost about 600 GB, against 160 GB, over about
    20 billion rows.
  - Never join periods on coordinates. Ruby's Mac differed from Linux in the 15th decimal on
    about 6,400 squares.
- `in_region` (ADDITION): the chip touches the unclipped subregions (AMW's "touch" rule,
  geo_filter.py:30). AMW keeps all chips of touching tiles.
- `in_andes`: the tile is in AMW's Andes tiling AND the chip touches the Andes boundary.
- Tile key format: `552:12:10.0:19:-15:-263` = size : pad : resolution : UTM zone : column : row.
  It sits on a fixed global DLTile grid, so it is the same in every period. `DLTile.from_key`
  rebuilds a tile with no network.
- **Still to design** (assemble/persist): period files, the final panels split by area so each
  chip's time series is together, and the `quarter` column in the monthly panel.

**What SAM2-later needs from the panel now** (all already planned):
- a stable chip_id plus the chip-square table
- the panel keyed by (chip_id, period)
- the prompt rule computable from columns (`det_loose`, `in_andes`, `confidence`)
- the frozen run config
- no dependence on detector imagery (AMW's SAM2 fetches its own 256 px tiles)

---

## 10. Variables still to build

**postprocess (assemble_period.py), per period across all subregions:**
- flags for loose and stringent: confidence ≥ t_main, and if the 5th nearest detected neighbour
  is beyond 3 km, it needs ≥ t_iso
- Andes: `in_andes` & confidence > 0.2, with no isolation rule (AMW adds Andes on top in
  `persistence.load_period`, keeping the higher confidence)
- Caveat: the neighbour count depends on how many chips were detected that period (cloud cover),
  and one scar fires several overlapping chips

**Persistence (persist.py).** AMW's actual rule, "Recipe A" (persistence.py:97-112):
- detected on the **annual** composite of year Y and again in Y+1 (k=2, window 2 years)
- from the loose set
- quarterly can't start or confirm a mine
- the newest year is provisional with t_prov 0.55

We have no annual composites, so the candidate variables (round2 §8) are:
- backward (safe as controls): `det_prev`, `n_det_before`, `first_det`, `periods_since_first`
- forward (outcome definitions only, clearly marked): `confirmed_k_n`, `onset`
- across frequencies: `det_in_quarter`

**Quality checks to implement:**
- `n_clear(Q) = n_clear(M1)+n_clear(M2)+n_clear(M3)` per pixel/chip. This is exact because the
  windows meet with exclusive ends, and a mismatch flags a date or data problem.
- `n_clear ≤ n_obs`.
- Assemble: every planned tile is scored or listed as failed. Failed tiles are **missing**, not 0.

**Econometric notes:**
- Include `n_obs`/`n_clear` and period fixed effects: satellite availability changed (Claude
  believes Sentinel-2B was lost around the end of 2021; unverified).
- Where two Sentinel-2 tiles overlap, one pass counts twice in both the counts and the median.
- Cluster standard errors (tile or Conley) because of the overlap.

---

## 11. Scale, storage, caching

- About 274,932 tiles × 529 = about 145 M chips. × 140 periods = **about 20 billion chip-period
  rows** and about 38 M Earth Engine requests.
- Each subregion holds about 34k–53k tiles.
- A full-ACA `plan.py` takes about 2–3 h and produces about 1.4 GB, once per run.
- Detect output is about 1 TB before compression (half of it the votes). An earlier, leaner
  estimate was about 360 GB.
- Detector imagery is about 8.6 MB per tile-period, about 200–300 TB in total, so it is **never
  cached long-term**. It stays in memory or on node-local disk during a job.
- Too many files is a risk: one file per tile would be about 38 M files. So we write part files
  per 500-tile batch, and assemble merges and deletes them.
- Storage tiers (cluster names are TBD by Ruby):

  | Tier | Holds |
  |---|---|
  | node-local `$TMPDIR` | the in-flight cache |
  | scratch (not backed up, file-count limits) | parts and period files during a run |
  | backed-up project storage | code, configs, plans, final panel, logs |
  | off-cluster | a copy of the final panel |

- Clearing: the node cache at job end; part files after assemble P passes its checks; period
  files after the final panel passes its checks.
- File format: **Parquet** (zstd), readable from Python, R and Stata (via Arrow).
- SAM2 later: about 15k tiles of 256 px per period (about 1–2% of the detector's pixels).
  Outputs about 0.6 GB per period with logits, about 80 GB total.

---

## 12. Open questions and pending decisions

1. **Persistence rule** (decide when writing persist.py):
   - (a) calendar version: a month in Y and a month in Y+1. Closest to AMW, but looser.
   - (b) rolling version: 2 detections within 12 months.
   - Or add **annual composites**: about 6% more Earth Engine requests, and it would let us copy
     AMW exactly and check against their published numbers.
2. SAM2 "backfill before" reading: never confirmed. It is deferred, so it isn't urgent.
3. Whether Claude should push small edits **straight to pipeline-rebuild**, instead of to its own
   branch. Asked on 15 Oct, no answer.
4. Whether to untrack `tests_output/` from git (Ruby's call).
5. The cluster's scheduler and storage names: Ruby will find these out herself, so don't ask.
6. Real Earth Engine behaviour: the request with the 2 count bands, timing and compute per tile.
   The first pilot measures this.
7. Proposed earlier but never implemented or decided: rounding pixels to whole numbers before
   scoring (round2 §3, to remove cache vs. fresh differences). This is moot without a cache.

---

## 13. Where the source thread stopped, and next steps

**Ruby's last message (19 Oct 06:39 UTC):**
> "now i want to actually do the first gee pull try. how do I do it?" and "when i run plan.py …
> I see only definitions … what is it that is actually running? … how does the computer know …
> the `--` after … where is this defined?"

**Answered by the source thread at 06:44 UTC.** Ruby hasn't reported back yet.
- **How a script runs:** written up in `notes/explainers/how_a_script_runs.md`.
  - The shell splits the line into words, and Python puts them in `sys.argv`.
  - `def` and `class` lines are executed, but they only *store* the function or class.
  - The work happens in the `if __name__ == "__main__":` block at the bottom, which is skipped
    when the file is imported.
  - `--options` mean nothing to the shell. Each script defines them with `ap.add_argument(...)`,
    and `ap.parse_args()` reads them (plan.py:158; detect.py around lines 306–311).
  - Unknown options exit with code 2; `--help` is automatic.
  - Pros and cons of the bottom-block style, and the `def main()` alternative, are discussed
    there. AMW uses the plain block (`core/inference_pipeline.py:99`).
- **First Earth Engine test, in the order given to Ruby:**
  1. `git pull origin claude/project-thread-ls8p9r` (on pipeline-rebuild). This brings
     `provenance.py`, the `earth_engine:` section and the new plan.py.
  2. Run `earthengine authenticate` (a browser login with the Google account that owns
     deforestation-and-mining; once per computer).
  3. A login check that downloads no imagery. It counts La Pampa images in July 2024 with
     `ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(ee.Geometry.Point(-69.9, -12.9)).filterDate('2024-07-01','2024-08-01').size().getInfo()`,
     after setting `gee.EE_PROJECT` from the config and calling
     `gee._ensure_earth_engine_initialized()`. The full one-liner is in the thread's 06:44
     reply. If Ruby reports "project not registered" or "API not enabled", that is the Cloud
     project setup, not our code.
  4. Delete `tests_output/pilot` and re-run `python panel_pipeline/plan.py`. The old config copy
     lacks `earth_engine:` and has `clip_to` twice.
  5. Once she has typed `detect.py` (from `notes/drafts/detect_walkthrough.md`; compare with
     `notes/drafts/detect.py`), the first real pull is one request:
     `python panel_pipeline/detect.py --run tests_output/pilot --frequency monthly --subregion 6 --from 2024-07 --to 2024-07 --limit 1`.
     La Pampa is all in subregion 6, and July is the dry season. Ruby was asked to send
     what it prints.

  What to check in her output, and what comes after it:
  - Inspect it:
     `python -c "import pandas as pd; d = pd.read_parquet('tests_output/pilot/detect/monthly'); print(len(d)); print(d.describe().T[['mean','min','max']])"`.
     Expect 529 rows; `n_obs` around 6 (up to about 12 in S2 tile overlaps); `n_clear ≤ n_obs`;
     `empty_share` near 0; `confidence` in 0..1.
  - Then score the full month (80 tiles, a few minutes) without `--limit`. A rerun should finish
     instantly (resume).
  - Packages: `python -c "import descarteslabs, pyarrow, sklearn, matplotlib"`. scikit-learn is
    needed by AMW's postprocess.py but missing from its requirements.txt.

**Then:**
- check Ruby's pushed detect.py
- run a small test area for 2–3 months through every step
- `assemble_period.py` (completeness check, concat + dedupe on chip_id with a dropped-rows count,
  postprocess variables, Andes flag)
- `persist.py` (decide §12.1)
- the final panels
- then Slurm/`submit.py`, then SAM2

---

## 14. Gotchas and superseded ideas

**Superseded (don't revive):**
- stage_review's first verdicts to drop concat, filter, postprocess and persist. Ruby overruled
  them: keep them as variables.
- Calling the end date a "bug". It isn't; it's a convention choice (§4).
- Splitting work by one deduplicated tile list instead of subregions. Ruby chose to keep AMW's
  subregions.
- Slurm shards and the `tasks.parquet` task table from run_design §4–5, and its older folder
  layout (`detect/<period>/<subregion>/shard_<n>/`). For the MVP the layout is §9 above. The
  routing principles still hold.
- Running SAM2 on the detector's 552 px tiles. That would be wrong: AMW prompts with one box
  around all detections per 256 px tile, so tile size changes the result.
- `imagery.ee_project`. It moved to `earth_engine.project`.
- The "2 detections in 12 months is AMW's rule" claim. It isn't (§10).

**Gotchas:**
- `sys.path` shadowing: AMW's `core/periods.py` vs ours. Always *append* `core/`, and import AMW
  modules after `grid`.
- An editor may underline AMW imports. Add `"python.analysis.extraPaths": ["core"]` to
  `.vscode/settings.json`.
- float32 `confidence` would flip `> 0.4` at exactly float32(0.4). Keep it float64.
- `True` counts as 1 in Python, so `_int` rejects bools.
- AMW's gee.py defaults to billing `earthindex`. Always set the project.
- AMW's postprocess uses `>=`, while its raw threshold uses `>`. Copy both exactly.
- AMW's cache stores rounded uint16 pixels (gee.py:493/501) while fresh pixels keep decimals.
  This is a likely cause of the small differences between duplicate copies.
- `plan.py` refuses an existing run folder. "already exists" on a second run is expected.
- Run commands from the repo root. Relative paths in the config resolve from the repo root.
- Never run two detect.py copies on the same (frequency, subregion). The duplicate guard will
  stop the second.

---

## 15. Source notes (in `/mnt/project-files/notes/`)

| File | What's there |
|---|---|
| `stage_review.md` | AMW's 13 stages from the code; how their driver and CLI work |
| `round2_answers.md` | overlap objections, concat, Andes check, votes, SAM2 design (§7), persistence variables (§8), scale (§9), producer/consumer + batch rewrite (§10) |
| `run_design.md` | job/thread vocabulary, the 4-script split, routing, storage tiers; dated update logs (8 Oct, 15 Oct ×2) |
| `explainers/how_a_script_runs.md` | what executes when a script starts, `__name__ == "__main__"`, `sys.argv`, argparse and `--help` (19 Oct) |
| `explainers/periods_explained.md` | classes, dataclass, field/property/method/classmethod, type hints, why `__future__` |
| `drafts/` | tested versions of every file: `config.py`, `config.yaml`, `periods.py`, `grid.py`, `plan.py`, `provenance.py`, `detect.py`, plus `detect_walkthrough.md` (block-by-block explanation + test plan) |
