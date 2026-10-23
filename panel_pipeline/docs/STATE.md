# Project state and decision log

Last full update: **23 Oct 2026**. This file replaces the handoff manual
(`history/handoff_manual.md`, 19 Oct) as the one place that says what is decided and where the
project stands. Keep it short and current: whoever changes a decision or finishes a step
updates the table here in the same commit.

What else to read: how to run anything is in `../README.md`; every open question is in
`open_design_decisions.Rmd`; measured results are in `findings.md`. Which document answers
what: `README.md` in this folder.

---

## 1. Goal and scope

- **Goal:** a clean econometric **panel** of Amazon Mining Watch's (AMW) mining-detector scores:
  one row per chip per period, with quality variables, built by a restartable pipeline.
  Nothing is published.
- **Base:** AMW's own inference code in `core/`, reused unchanged wherever possible. The
  `timeseries/` attempt is not used (two Earth Engine calls per tile).
- **Periods:** months and quarters, Jan 2018 to Sep 2026: 105 months + 35 quarters = 140
  periods. No annual periods.
- **Region:** AMW's **Amazon_ACA**, 6 subregions, including the Andes (threshold 0.2, AMW's
  logic). 274,932 tiles, about 145 M chips, about 20 billion chip-period rows.
- **Compute:** the Goethe cluster in Frankfurt (CSC). Earth Engine on the partner tier, Google
  Cloud project `deforestation-and-mining`.
- **Scope now:** scores, AMW's postprocess flags, then persistence as flags. **SAM2 later**,
  as a separate step that adds its own columns.

## 2. How Ruby works (for anyone helping)

1. Ruby types the main code herself, from code given block by block with a line-by-line
   explanation. Small fixes (comments, config, helper moves, paths) can be pushed to a branch
   for her to pull. On 23 Oct she asked for `assemble_period.py` and the folder restructure to
   be pushed directly.
2. Explain the computer-science concepts behind every step, with rationale and honest
   objections. She wants to think like a senior developer.
3. Don't assume; ask when something about the project is unknown.
4. **Read AMW's code, not its comments** (often outdated). Cite `file:line`.
5. Stay as close to AMW as possible. Mark every reuse and change in the code:
   `# AMW: file.py:line`, `# Same as AMW: ...`, `# DEVIATION from AMW: what. Why.`,
   `# ADDITION: ...`. Then `grep -rn "DEVIATION\|ADDITION" panel_pipeline/` is the methods list.
6. MVP first, run by hand; scheduler scripts only once the MVP works.
7. Settings go in `config.yaml`, each topic in its own section; no environment variables.
8. Style: `from __future__ import annotations` first; built-in `list[...]`/`dict[...]` hints.

## 3. Where things stand

| Step | Code | Run so far |
|---|---|---|
| `plan.py` | done | test_region, test_region_large, test_region_para, amw_compare_s3 (all of subregion 3) |
| `detect.py` | done; streaming, 2-min saves, `_log.txt`, Ctrl-C saves, 300 s timeout, 32-bit pixels, compiled scoring (PR #1, merged 23 Oct) | test_region 2026 monthly + quarterly; Pará 2026Q1–Q3; **all of subregion 3, 2026Q3** (34,825 tiles, 0 failed) |
| `status.py` | done, with `--todo` | |
| `assemble_period.py` | **first version** (23 Oct): layers `scores` and `postprocess` | tested on the two small test runs; flags equal AMW's `dual_threshold_filter` in all 24 test periods |
| `build_panel.py` (persistence, across time) | not started; rule open (Rmd D8) | |
| Cluster | `cluster/` job scripts and README Part B written (23 Oct) | not run on Goethe yet |
| SAM2 | deferred | |

**Before the full run, still open:** the Earth Engine compute budget and the scope of periods
(`cluster/cluster_plan_goethe.md` R2, R5; at about 90k tile-periods a day the full plan takes
about 14 months), the service-account `credentials:` key in `lib/config.py`
(`cluster/service_account_guide.md` Part C), and the cell-structure questions (Rmd D5, D6).

## 4. Decision log

Newest decisions first within each topic. "Rmd Dn" = `open_design_decisions.Rmd`.

| Topic | Decision | Who, when |
|---|---|---|
| What we keep from AMW | Inference engine; concat/dedupe; postprocess and persistence **as variables**; SAM2 later. Dropped: cog, persist-masks, stage, manifest, publish | Ruby, 8 Oct |
| Panel is complete and additive | Every planned chip has a row in every period; nothing is filtered or deleted; later steps only add layers. Failed tiles are missing (`scored = FALSE`), never 0 | Ruby, 21 Oct (Rmd P1, P2, D11) |
| Panel unit | AMW's overlapping 48 px chips, 24 px stride, 529 per tile. A non-overlapping cell is still open (Rmd D5, D6: 240 m cells proposed) | Ruby, 8 Oct; open since 19 Oct |
| Every chip written | Every chip's score, not just positives (AMW keeps only > 0.4) | Ruby, 8 Oct |
| Ensemble votes | All 6 votes as float32 `vote_1..6`; `confidence` = their mean, **float64** | Ruby, 8 Oct |
| Andes | No second inference run; `in_andes` flag, threshold 0.2 applied in assembly | 8 Oct |
| Periods | Separate Earth Engine composites per month and per quarter; monthly and quarterly are separate panels joined on `chip_id` | Ruby, 8 and 12 Oct |
| **Last day of each period** | Period ends on the 1st of the next period (excluded by Earth Engine), so **every day is imaged**. AMW passes the last day as the end and so drops 31 Mar, 30 Jun, 30 Sep, 31 Dec. Proven, see `findings.md` §3. Keep ours | Ruby, 20 Oct (Rmd D9 keeps the alternative on record) |
| Cloud recipe | AMW's median with Cloud Score+ `cs_cdf ≥ 0.6`, same for months and quarters | Ruby |
| Quality variables | `n_obs`, `n_clear`, `empty_share` per chip, from extra bands in the **same** Earth Engine request | 15 Oct |
| Work unit | One task = one **frequency × period × subregion** folder (840 tasks for the full plan); a task never shares its folder (lock) | Ruby, 19 Oct |
| Lock | `_lock.json` per task folder with a 1-minute heartbeat; stale after 15 min; never run two processes on one subregion-period | 19 Oct |
| Saving | Downloads stream without pause; a part is saved every 2 min or every `--batch-tiles` tiles; a finished task is merged into `part_00000.parquet` | 20 Oct (PR #1) |
| Downloads | 32-bit pixels (18.6 MB a tile instead of 34.5 MB, identical values); 300 s timeout per request | 21–22 Oct (PR #1) |
| Scoring on a Mac | Compiled ("graph") TensorFlow, the same as AMW's Linux machines and the cluster; `--mac-eager` restores AMW's Mac setting | Ruby, 20 Oct (PR #1) |
| `filterBounds` recipe | Not used: identical pixels but no reliable saving | 22 Oct |
| Run-time overrides | `--workers`, `--batch-tiles`, `--flush-minutes`, `--tries` may differ from the frozen config; anything that changes a score needs a new run | 19–20 Oct |
| Image cache | None for real runs (200–300 TB). `--image-cache` exists for test runs only; it writes as well as reads | 15 and 19 Oct |
| Config | One `config.yaml`, checked by `lib/config.py`, **frozen into the run folder** by plan.py; later steps read only `<run>/run/config.yaml` | Oct |
| Earth Engine project | Own section `earth_engine: project:`; no default (AMW's default bills `earthindex`) | Ruby, 15 Oct |
| Run naming | `<area>_<YYYY-MM-DD>`; config in `configs/<name>.yaml`, committed; test output in `tests_output/` (git-ignored) | 19 Oct |
| Region | Amazon_ACA, not `amazon_basin` | Ruby, 8 Oct |
| Assembly | Two scripts split by axis: `assemble_period.py` (one period, across space) and `build_panel.py` (one area, across time) | Ruby, 21 Oct (Rmd P5) |
| Layers | One file per layer and period, every planned chip once, in `chip_id` order: `<run>/panel/<freq>/<layer>/period=<tag>/data.parquet` + `_record.json` | 21 Oct (Rmd D13) |
| One run at a time | The whole region is one run; parallel work happens inside it | Ruby, 21 Oct (Rmd P4) |
| Tools | pandas and pyarrow; DuckDB open | Ruby, 21 Oct (Rmd P6, D10) |
| Border tiles | Left out of the MVP: assemble refuses a period with a tile listed twice (Rmd D7) | Ruby, 21 Oct |
| Distances | Undecided; the MVP stores AMW's distance and a geodesic one (Rmd D3) | 21 Oct |
| postprocess | AMW's loose (0.43/0.75) and stringent (0.55/0.8) rules, 5th neighbour beyond 3 km, as **flags** | Ruby |
| Persistence | A flag; the rule is open (Rmd D8). `config.yaml` holds k = 2 within 12 months as a placeholder | Ruby |
| Cluster shape | One job on one node, 2 tasks side by side with ~20 workers each (= 40 requests in flight, Earth Engine's default), resubmitted; more nodes don't help | 23 Oct (`cluster/cluster_plan_goethe.md` R2) |
| Folder layout | Steps at the top of `panel_pipeline/`, helpers in `lib/`, checks in `checks/`, notes in `docs/` | Ruby, 23 Oct |

Superseded and not to revive: dropping concat/postprocess/persistence (stage_review);
calling the end date a "bug"; Slurm shards and `tasks.parquet` (run_design); 500-tile batches
that pause the downloads; deleting part files after assembly; running SAM2 on the detector's
552 px tiles; `imagery.ee_project`; "2 detections in 12 months is AMW's rule" (it isn't: AMW
uses annual composites, year Y and Y+1).

## 5. AMW code we import

AMW's `core/` is put on Python's search path by `lib/grid.py` (appended, so AMW's
`core/periods.py` never shadows ours). **Don't move or rename these files.**

| AMW file | What we use | Used in |
|---|---|---|
| `core/tile_utils.py` | `create_tiles`, `cut_chips` (and `CenteredTile`, `ensure_tile_shape` via gee.py) | `lib/grid.py`, `detect.py` |
| `core/gee.py` | `GEE_Data_Extractor` (subclassed as `PanelExtractor`, only `_build_composite` overridden), `DataConfig`, `EE_PROJECT`, the login | `detect.py` |
| `core/postprocess.py` | `centroid_key`; in assembly also its dual-threshold and neighbour functions | `lib/grid.py`, `assemble_period.py` |
| `core/tf_darwin.py` | `apply_darwin_tf_compat`, **only with `--mac-eager`** since 20 Oct | `detect.py` |

Copied rather than imported (cited in comments): the four scoring lines and the tries loop of
`core/inference_engine.py` (importing it would pull in torch and SAM2). Don't import
`core/persistence.py` either: it imports `periods` and would find ours. `postprocess.py`
needs scikit-learn, which AMW's requirements leave out (it is in `requirements/`).

Used as data: the model `models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` (sha256 pinned per
run), `data/boundaries/Amazon_ACA/Amazon_ACA_{1..6}.geojson`,
`data/boundaries/andes_supplemental.geojson`, and the test areas.

## 6. Data layout

```
<output_root>/<run name>/
  run/config.yaml, run/provenance.json        frozen settings; code, packages, fingerprints
  plan/tiles.csv                              one row per (subregion, tile); border tiles twice
  plan/chips.parquet                          chip_id, tile_key, chip_index, square, amw_key_x/y, in_region, in_andes
  detect/<freq>/period=<tag>/subregion=<n>/
      part_00000.parquet                      merged when the task is complete (part_0000N while running)
      _status.json  _lock.json  _log.txt
  panel/<freq>/<layer>/period=<tag>/
      data.parquet  _record.json              layers: scores, postprocess (more later)
```

- detect columns: `chip_id`, `confidence` (float64), `vote_1..6` (float32), `n_obs`,
  `n_clear`, `empty_share`. Hive folders, so pandas adds `period` and `subregion`.
- `chip_id` (int64) = UTM zone | tile column + 500 | tile row + 5000 | chip index (column-major,
  AMW's `cut_chips` order). Never join on coordinates (Mac and Linux differ in the 15th decimal).
- Tile key `552:12:10.0:<zone>:<col>:<row>`: a fixed global grid, the same in every period.

## 7. Facts that were only in thread messages

- On a Mac, only `--workers` adds speed. Never start several processes on one subregion-period:
  they would overwrite each other's part files (the lock refuses the second).
- `--from`/`--to` include both ends. One subregion per command; loop over subregions in the shell.
- A closed laptop lid freezes a run; use `caffeinate -i` and keep the lid open. The 300 s
  timeout makes a frozen download fail and retry instead of hanging.
- Every task folder has files with the same names by design: the folder path is the identity.
- chip_ids are computed from tile and chip position, not counted, so every run gives a chip
  the same id.
- Border tiles appear under two subregions with the same chip_ids (1,186 tiles in ACA).
- "One run, many tasks": 840 detect tasks and 140 assemble tasks; jobs are not nodes.
- AMW measures distances in one UTM zone (geopandas default); about 5% off at ACA's edges.
- Earth Engine adds scenes late; re-score the last quarter before the final panel (Rmd
  section "Leftovers", C1).
- Service-account key files never go in git (`.secrets/`, `deforestation-and-mining-*.json`
  are in `.gitignore`).
