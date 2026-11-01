# Panel assembly: strategy for the two scripts (draft for discussion)

Written 19 Oct 2026 for Ruby. Nothing here is built yet. It starts from the plan in
`handoff_manual.md` §6 and §10 (`assemble_period.py`, then `persist.py`), checks it against
the code on `pipeline-rebuild` (head `f349915`) and against your two test runs, and says where
I'd keep it, where I'd change it, and what I need you to decide. Numbers marked *measured*
come from your runs; the rest are estimates.

## Update 21 Oct: Ruby's decisions, and what changed

Decided (the full list, with every open item, is now in the repo:
`panel_pipeline/docs/open_design_decisions.Rmd`):
- Two scripts split by axis: **yes**. Balanced panel: **yes**. One run at a time: **yes**
  (parallel tasks *inside* a run, split by period and subregion).
- **The panel is complete and additive.** No rows are ever dropped, nothing is deleted, and
  later steps only add variables and flags. Persistence is a flag. This overrides §3 step 7
  and the "delete parts after assembly" plan in the handoff and cluster notes (Rmd D11).
- **pandas and pyarrow** for now; DuckDB moved to the cluster migration thread (Rmd D10).
- Border tiles: left out of the MVP, kept modular (Rmd D7).
- Distances: undecided (Rmd D3). The MVP stores AMW's distance and a geodesic one.
- The isolation, edge, blob, seam and unit objections: agreed, handled later (Rmd D1-D6).
- Last-day convention added as an open decision (Rmd D9).

Changes to the design below:
- **Layers.** Each step writes its own layer, one file per period, holding every planned chip
  once in chip_id order, so layers line up row for row. Adding a variable = adding a layer;
  nothing existing is rewritten. Layout:
  `<run>/panel/<frequency>/<layer>/period=<tag>/data.parquet` plus `_record.json`.
  `assemble_period.py` writes the layers `scores` and `postprocess`.
- **Memory.** A Goethe node has 192 GB, so assemble holds one period whole (~10-20 GB for all of
  Amazon_ACA) instead of streaming. Simpler code; a laptop can only assemble test runs.
- **Drafts (tested on both test runs):** now in the repo as `panel_pipeline/assemble_period.py` and
  `panel_pipeline/lib/read_panel.py`. The postprocess flags match AMW's own `dual_threshold_filter`
  chip for chip in all 24 test periods.

---

Contents:
1. What your two test runs can and cannot test
2. The core idea: the data is a table of chips × periods, too big to hold
3. Script 1, `assemble_period.py`: across space, one period at a time
4. Script 2, `build_panel.py` (was `persist.py`): across time, one area at a time
5. Statistics: what the panel should contain, and my objections
6. Scalability and robustness
7. Build order and how we work
8. Decisions for you

---

## 1. What your two test runs can and cannot test

| | Run A: `test_region_2026-10-19` | Run B: `test_region_para_2026-10-19` |
|---|---|---|
| Where | Venezuela/Guyana, ~5°N 62.5°W, UTM zone 20 | Pará, ~8°S 55°W, UTM zone 21 |
| Subregion | 2 only | 4 only |
| Tiles / chips | 36 / 19,044 | 42 / 22,218 |
| Periods | monthly 2026-01…09, quarterly 2026Q1…Q3, all complete | quarterly 2026Q1, Q2 complete; Q3 was being scored at 12:40 |
| Detections > 0.4 (Q1 / Q2) | 84 / 105 | 56 / 732 |

**They can test:** reading and checking the parts, the completeness gate (run B's Q3 is a
real "not finished yet" case), the period files, the isolation variables, monthly ↔ quarterly
joins (run A), the time variables over 9 months, and the area-split panel.

**They cannot test** four things the full run will hit:
- **Border tiles.** Each run lies in one subregion, so no tile is listed twice
  (`tiles.csv`: 0 duplicates in both). The dedupe step would never fire.
- **Andes.** 0 Andes chips in both runs.
- **UTM zone seams.** Each run lies inside one zone.
- **More than one subregion per period**, which is the whole point of "assemble".

Also, the two runs are **two separate runs**: two plans, two frozen configs, two chip
universes. In our design a run is one `plan.py` result, and the real run will be one run for
all of Amazon_ACA. So "across regions" inside the real pipeline means *across subregions of one
run*, not *across runs*. My recommendation (decision 5): each script works on one run;
combining runs is not a pipeline step. To test the cross-subregion logic for real, plan **one
small extra test run that straddles a subregion border, a UTM seam and the Andes edge**
(decision 6), plus a synthetic test that copies a few tiles into a second subregion folder.

---

## 2. The core idea: a table too big to hold, cut along two different axes

Think of the finished data as one giant spreadsheet: one row per chip (~145 M for
Amazon_ACA), one column-block per period (140). That's ~20 bn chip-period rows (handoff §11).
A full period in pandas is ~18 GB (*measured* 124 bytes/row), the whole thing ~2.5 TB. No
laptop or single cluster job holds it.

The computations we need come in two kinds, and they need **different slices**:

| Computation | Needs | Slice |
|---|---|---|
| Remove border duplicates, isolation rule (5th neighbour within 3 km) | all chips of **one period**, across subregions | a **period** (a column of the spreadsheet) |
| `det_prev`, first detection, persistence, monthly ↔ quarterly | all periods of **one chip** | an **area** (a block of rows) |

That's why there are two scripts, and why they're in this order. Detect writes the data
sliced by (period, subregion). Script 1 reads one period, finishes everything spatial, and
writes it out **sorted by `chip_id`**. Script 2 then reads one *area* (a `chip_id` range) from
every period file and finishes everything temporal.

**Computer science concept: the "transpose" (or "shuffle").** Going from period slices to area
slices is the expensive step in any big-data job, because every output file needs a little
piece of every input file. We make it cheap with one trick: Parquet files are stored in **row
groups** (blocks of rows), and each row group records the min and max of every column. If the
period files are sorted by `chip_id`, a reader asking for "chip_ids between X and Y" opens only
the row groups that can contain them and skips the rest. So script 2 reads 1/N of each period
file instead of all of it. That's the single design choice that makes the panel step scale.

`chip_id` order is zone → tile column → tile row → chip (handoff §9), and `chip_id // 1000`
is the tile (detect.py:193). So "an area" is simply a run of consecutive tiles: a vertical
strip inside one UTM zone.

---

## 3. Script 1: `assemble_period.py` (space, one period at a time)

```
python panel_pipeline/assemble_period.py --run tests_output/<run> --frequency monthly --period 2026-01
```

One job = one (frequency, period). Steps:

1. **Completeness gate.** For every subregion in the run's `tiles.csv`, read its
   `_status.json`. Refuse unless every one is `complete` (every tile saved or failed after all
   tries). Also refuse if their `model_members` differ. Failed tiles are allowed but listed.
   *Why:* the isolation rule counts neighbours across subregions, so a missing subregion
   silently makes its neighbours look isolated. Assembling half a period is worse than not
   assembling it.
2. **Read** the parts of every subregion, plus the chip attributes from `plan/chips.parquet`
   (`tile_key`, `in_region`, `in_andes`, the square). Only the columns needed.
3. **Dedupe border tiles** on exact `chip_id`, keeping the copy with the higher confidence and
   its votes (AMW: `postprocess.py:146`, `dedupe_detections`; DEVIATION: key = `chip_id`, not
   rounded centres). Count dropped rows. Only tiles listed under 2+ subregions in `tiles.csv`
   can be duplicates (1,186 of 274,932), so this is a tiny side-job, not a pass over everything.
4. **Balance.** Every planned chip gets a row. Chips of failed tiles get `scored = False` and
   missing (NaN) scores. (Decision 4.)
5. **Spatial variables** (§5): `det_raw`, `det_andes`, and for loose and stringent the
   catalogue's `kth_neighbor_km` and the `det_loose` / `det_stringent` flags, computed on the
   whole period's catalogue at once with AMW's `kth_neighbor_km_on_catalog` (postprocess.py:210)
   and `keep_dual_threshold` (:231). The catalogue is only chips ≥ t_main, so it's small
   (run B Q2: 701 of 22,218 chips), even for the whole Amazon.
6. **Write** `assemble/<frequency>/period=<tag>.parquet`, sorted by `chip_id`, row groups of a
   fixed number of tiles, plus `period=<tag>.json`: inputs, row counts, duplicates dropped,
   failed tiles, catalogue sizes, the period's mean `n_clear`, git commit. Temp file then
   rename, as in detect (detect.py:211-218).
7. **Don't delete the parts** in the MVP. Deleting becomes a separate, explicit step once the
   checks have passed (on the cluster: after the period is archived).

**Memory.** Steps 1-5 never need the whole period at once if the heavy reading and sorting is
done by a tool that streams from disk (decision 2). Only the catalogue (detections) and the
border tiles are held in memory.

---

## 4. Script 2: `build_panel.py` (time, one area at a time)

The handoff called this `persist.py`. It does more than persistence, so I'd rename it.

```
python panel_pipeline/build_panel.py --run tests_output/<run> --frequency monthly --block 3
```

One job = one (frequency, area block). A block is a range of consecutive tiles, sized so one
block's full time series fits comfortably in memory (e.g. 200 tiles × 529 chips × 105 months
≈ 11 M rows ≈ 1.4 GB). Full ACA: ~1,400 blocks per frequency, all independent, so they run as
a cluster job array. On your test runs everything is one block.

Steps:
1. Refuse unless **every** period of that frequency is assembled.
2. Read this block's `chip_id` range from every period file (row-group skipping, §2).
3. Compute the per-chip time variables (§5.3), ordered by `month_index` (periods.py:43).
4. Monthly only: add `quarter`, and join that quarter's key columns from the quarterly panel.
   Quarterly only: add summaries of its three months.
5. Write `panel/<frequency>/block=<b>.parquet` (Hive layout, so `read_parquet("panel/monthly")`
   reads all blocks as one table) and a record.
6. Checks: rows = chips × periods; one row per (chip_id, period); the `n_obs` / `n_clear`
   identity quarter = sum of its 3 months (exact in your run A, *measured*).

---

## 5. Statistics: what goes in the panel, and my objections

### 5.1 Principle: store the continuous measures, derive the flags

Every flag must be reproducible from columns in the same row. So we store `confidence`, the 6
votes, `kth_neighbor_km`, and the image counts, and the flags next to them. Then a referee
asking "what if the threshold were 0.5" is a one-line recomputation, not a rerun.

### 5.2 Columns per (chip, period)

| Group | Columns | Source |
|---|---|---|
| Keys | `chip_id`, `period`, `month_index`, `subregion` (after dedupe: the one kept), `tile_key` | detect + plan |
| Location | `in_region`, `in_andes`, zone (from chip_id), later the cell id from the cell thread | plan |
| Score | `confidence`, `vote_1..6`, **`vote_sd`** (ADDITION: disagreement between the 6 models, a cheap uncertainty measure) | detect |
| Image quality | `n_obs`, `n_clear`, `empty_share`, `scored` | detect |
| Detection | `det_raw` (> 0.4), `det_andes` (`in_andes` & > 0.2), `det_loose`, `det_stringent` | assemble |
| Context | `kth_km_loose`, `kth_km_stringent` (NaN if not in that catalogue), `near_edge` (within 3 km of the run's outer edge) | assemble |

Plus one small **period table** (one row per period): catalogue sizes, mean `n_clear`, share of
chips with `n_clear < 1`, failed tiles. That's your period-level control and the first place a
bad month shows up.

### 5.3 Time variables (script 2)

Backward-looking (safe as controls, handoff §10): `det_prev`, `n_det_before`, `first_det`,
`periods_since_first`, `n_clear_prev`. Forward-looking (outcome definitions only, clearly
named): `confirmed_k_w` and `onset`. **The persistence rule is still open** (handoff §12.1:
calendar Y/Y+1, rolling 2-in-12, or annual composites). I'd build the backward variables first
and leave the rule as a parameter. Your test runs only cover 2026, so the calendar version
can't be tested on them at all.

### 5.4 Objections you should hear before we copy AMW's postprocess

1. **The isolation rule depends on cloud cover, measurably.** In run B, Q1 (mean `n_clear`
   4.9) has 51 catalogue chips and **27 of them are isolated**; Q2 (`n_clear` 19.5) has 701 and
   **1 is isolated** (*measured*, haversine distances). Fewer clear images → fewer detections →
   fewer neighbours → more chips fail the stricter t_iso. So `det_loose` mixes "is there a
   mine" with "was it cloudy". As an *outcome* in a regression that's a measurement error
   correlated with season and region. Consequences: always control for `n_clear` and period
   fixed effects; use `det_raw` + controls as the main outcome and `det_loose` as robustness,
   or the reverse with a stated reason. Keeping `kth_neighbor_km` as a column lets you test it.
2. **Edges look isolated.** A chip near the edge of the scored area has neighbours only on
   one side. In run A Q1, 4 of the 14 isolated chips are within 3 km of the test area's edge
   (*measured*). In the full run this only happens at Amazon_ACA's outer border, but in every
   *test* run it's everywhere, so test-run isolation numbers aren't representative. Hence the
   `near_edge` flag.
3. **AMW measures distance in one UTM zone for the whole basin.** `centroids_xy_m`
   (postprocess.py:190) projects every chip into the single zone estimated from the file's
   centre. Amazon_ACA spans 7 zones (17-23), so the far west and east sit 17-19° of longitude
   from that zone's central meridian, where distances come out about 4-5% too long (UTM scale
   error ≈ ½·(Δλ in radians)² near the equator). "3 km" then means about 2.85 km there. Proposal (DEVIATION,
   decision 3): great-circle (haversine) distances with a ball tree, accurate everywhere, same
   `k`, same 3 km. On your single-zone test runs both give practically the same answer, so the
   deviation is cheap to verify.
4. **One mine fires a blob of chips.** Chips overlap 4× (stride 24 px), so a scar of a few
   hundred metres lights up a 2×2 or 3×3 block, and its 5th "neighbour" is often itself:
   31-61% of catalogue chips have their 5th neighbour under 0.5 km (*measured*). The rule is
   AMW's and we keep it, but read "clustered" as "more than a tiny blob", not "near other mines".
5. **Zone seams double-count area** (cell-structure thread). A ~9 km strip at every seam is
   scored twice on two grids, and no dedupe catches it. **This is open in the "Mutually
   exclusive cells for the panel" thread and I won't decide it here.** The design keeps it
   pluggable: whatever rule is chosen becomes a column in the chip table (e.g. `keep` or
   `cell_id`), and both scripts carry chip attributes through. So it doesn't block building,
   but it must be decided before the full run's panel is used for areas or counts.
6. **The unit question is still open too.** If your regressors vary by municipality or
   concession, the chip panel is the measurement layer and a third, small step aggregates it to
   that unit. That step is cheap once the chip panel exists; it just shouldn't be confused with
   these two scripts. The cell thread asked you which unit; that answer decides step 3.

### 5.5 Missing is not zero

A failed tile or a chip with no clear image must never become "no mine". Balanced panel with
`scored = False` for failed tiles (decision 4); low-quality chips keep their score, flagged by
`n_clear` / `empty_share`, and the analysis decides the cut. I would *not* bake a "usable"
threshold into the panel: that's a modelling choice that belongs in the analysis script, where
it's visible.

---

## 6. Scalability and robustness

| Concern | Design |
|---|---|
| Memory | Never load a full period (assemble) or a full panel (build). Streaming reads, small catalogue in memory, blocks sized by config |
| Parallelism | Assemble: one job per (frequency, period), 140 jobs. Build: one per (frequency, block), ~2,800 jobs. No job writes another's file, same as detect |
| Order | Assemble a period as soon as its subregions are complete (period-first order, already decided). Build waits for all periods |
| Restart | Idempotent: an output whose recorded inputs are unchanged is skipped; reruns redo only what's missing. Temp file then rename. Reuse `lock.py` so two copies can't write the same output |
| Files | Assemble turns ~92 parts × 6 subregions into one file per period; build writes ~1,400 files per frequency |
| Provenance | Every output gets a JSON record: inputs (and their `_status.json` session times), counts, git commit, environment (reuse `provenance.py`) |
| Progress | Extend `status.py` with two more columns: assembled? built? |
| Disk | Parts deleted only by an explicit command after checks pass and (cluster) after archiving |

**Engine (decision 2).** Two options for the heavy reading, joining and sorting:
- **DuckDB** (recommended): an embedded SQL database that reads Parquet directly, sorts and
  joins tables bigger than memory by spilling to disk, and does per-chip time variables with
  one window function (`LAG(det_raw) OVER (PARTITION BY chip_id ORDER BY month_index)`). It's a
  `pip install`, no server. For an economist SQL reads a lot like Stata's `by chip: gen`.
  Objections: a second language to learn, and errors come back as SQL messages.
- **pandas/pyarrow only**: what you know already. Works for the test runs as is, but for the
  full run we'd hand-write the streaming and the out-of-memory sort, which is exactly the kind
  of code that has subtle bugs.

Either way the AMW parts (kNN, dual threshold, dedupe rule) stay in Python with numpy and
scikit-learn.

---

## 7. Build order and how we work

As before: you type the main code; I give it block by block with the explanation, keep a
tested draft in `notes/drafts/`, and check your pushes.

1. `assemble_period.py` on run A, 2026-01 (smallest real case), then all of run A and run B.
2. A synthetic border test: copy a few of run A's tiles into a fake `subregion=3` folder with
   slightly changed scores, and check that the dedupe keeps the higher one and counts them.
3. `build_panel.py` on run A, monthly and quarterly.
4. The extra straddling test run (decision 6), through both scripts.
5. Then the cluster wrappers, together with detect's.

---

## 8. Decisions for you

1. **Two scripts split by axis** (space per period, then time per area). Recommended: yes.
2. **Engine:** DuckDB (recommended) or pandas/pyarrow only.
3. **Distances:** haversine (recommended, a DEVIATION) or AMW's single UTM zone.
4. **Balanced panel**, with `scored = False` rows for failed tiles. Recommended: yes.
5. **One run at a time** (recommended) or a merge across runs. Did you want the two test runs
   combined into one panel, or were they meant as two separate tests?
6. **An extra small test run** across a subregion border, a UTM seam and the Andes edge.
   Recommended: yes; I'd pick the area and you'd score 2-3 quarters.

Open elsewhere, not blocking the build: the cell structure / zone seam rule and the analysis
unit (cell-structure thread), and the persistence rule (handoff §12.1).
