> **History, not maintained.** 8–15 Oct. Shards, `tasks.parquet` and its folder layout were dropped; the layout is in `../../README.md` and `../STATE.md`. The routing principles still hold.

# How the pipeline runs: stages, jobs, routing, storage

Builds on [round2_answers.md](round2_answers.md). Scope for now: everything up to
and including the persistence variables. SAM2 comes later.

---

## 0. New decisions recorded

| Topic | Decision |
|---|---|
| SAM2 | Deferred. Built later as a separate step that adds its own columns (§1) |
| Periods | Jan 2018 to Sep 2026: 105 months + 35 quarters = 140 periods |
| Branch | `pipeline-rebuild` (it contains both `code-walkthrough` and `main`) |
| Earth Engine | Partner tier. We still cap simultaneous requests (§4) |
| Persistence | k = 2 detections within 12 months, both settable in config |

**Something I found on `pipeline-rebuild`:** the boundary files the old pipeline
uses (`Amazon_ACA.geojson`, the six `Amazon_ACA_1..6.geojson` and
`andes_supplemental.geojson`) are **not on the branch**. They were removed in the
archive commit `e15dae2`. What remains is `amazon_basin.geojson` and
`amazon_1..6`, which is a different, smaller area. I measured both:

| | Area | Andes area inside it |
|---|---|---|
| `Amazon_ACA` (their 2025–26 production runs) | 8.24 million km² | 100% |
| `amazon_basin` (on the branch) | 6.70 million km² | 95.4% |

The two differ by 1.55 million km². This is a decision for you (question 1 at the end).

### Update 2026-10-08 (MVP decisions)

- **MVP first, run by hand.** No Slurm yet. Scripts are started by hand, stage by stage.
- **Work unit = theirs:** one process per (period, subregion), with their 500-tile
  batches inside it. Each batch writes its own part file instead of rewriting the
  whole output. Shards and Slurm arrays come later.
- **Region: Amazon_ACA** (their production area), restored from commit `bac9d30`.
- **One cluster**, several machines. The cluster's storage names are Ruby's to fill in.
- **Period tags:** `2018-01` (month), `2018Q1` (quarter). End dates exclusive (1st of next month).
- Tested drafts of agreed code are kept in `notes/drafts/`.
- **Frequencies never mix.** Monthly and quarterly are separate runs, separate folders
  (`<RUN>/monthly/...`, `<RUN>/quarterly/...`) and two separate panels with the same chip_id.
  They are linked only by a `quarter` column in the monthly panel, in a final step.
- **Run order (Ruby):** all monthly periods for subregion 1, then subregion 2, ..., then quarterly.
  detect takes `--frequency --subregion --from --to`.
- Type hints use built-in `list[...]`, not `typing.List`.

### Update 2026-10-15 (plan.py; tested drafts in `notes/drafts/`)

- **Reuse AMW code wherever possible**, and mark every line in comments:
  `# AMW: file:line` (reused or copied), `# Same as AMW:` (kept on purpose),
  `# DEVIATION from AMW: what. Why.`, `# ADDITION:` (new, nothing dropped).
  `grep -rn "DEVIATION\|ADDITION" panel_pipeline/` gives the methods-section list (Ruby's request).
- **plan.py writes the run's config copy, once.** Later scripts read `<run>/run/config.yaml`,
  never `panel_pipeline/config.yaml`. `run/provenance.json` holds git commit, uncommitted
  files (config excluded), package versions and sha256 of the model and boundary files;
  detect.py should refuse to score if the model's sha256 differs.
- **Run folder:** `<output_root>/<name>/run/{config.yaml,provenance.json}` and
  `plan/{tiles.csv,chips.parquet}`; built as `<name>.planning/`, renamed when complete;
  an existing run folder is never re-planned.
- **grid.py** (space axis, like periods.py for time) wraps AMW's `create_tiles`,
  `cut_chips` (blank image gives AMW's exact chip squares) and `centroid_key`.
  core/ is APPENDED to sys.path because AMW's `core/periods.py` would shadow ours.
- **chip_id** = int64 `zone | ti+500 | tj+5000 | chip index`, e.g. 194784737000
  (zone 19, tile -22:-263, chip 0). Chip index = AMW's cut_chips order (column-major).
  Amazon_ACA range: zones 17-23, ti -61..60, tj -411..201.
- **tiles.csv** is long: one row per (subregion, tile_key); border tiles listed under each
  subregion, exactly like AMW (1,186 border tiles; 274,932 unique; 276,120 listed).
- **chips.parquet** columns: chip_id, tile_key, chip_index, minx/miny/maxx/maxy (full
  precision), amw_key_x/y, in_region (touches the unclipped subregions, ADDITION),
  in_andes (tile in AMW's Andes tiling AND chip touches the Andes boundary, AMW's rule).
  All 390 Andes tiles lie inside the subregion lists.
- **config.yaml `region.clip_to`** (test runs only, null = whole region); first test area is
  AMW's `la_pampa_corridor` (80 tiles, 42,320 chips). Madre de Dios AOI: 748 tiles, 32,101 Andes chips.
- **Persistence: AMW's rule is NOT 2-in-12-months.** It is k=2 within window=2 *years* on
  annual composites only (detected in year Y and Y+1; persistence.py:97-112), from the loose
  set, with t_prov 0.55 for the newest year. Open for persist.py: (a) month in Y + month in Y+1,
  (b) rolling 2 in 12 months, or add annual composites (~6% more EE requests) to copy it exactly.
- AMW's `postprocess.py` imports scikit-learn, which AMW's requirements.txt does not list.
- **Period end dates:** ours (1st of next period, excluded) matches AMW's earlier runs and
  training imagery (`..._2024-01-01_2025-01-01`); AMW's current driver passes the last day
  (pipeline.py:80-81,160 via core/periods.py:84), so it drops that day. Ruby corrected my first claim.
- **Earth Engine billing:** AMW's gee.py:34 bills `EE_PROJECT`, defaulting to upstream's
  `earthindex`. detect.py will take an explicit `ee_project` from config (no default). Asked Ruby
  for her partner-tier project (2026-10-15). No EE calls have been made from Claude's container.
- **Claude's edits** go to branch `claude/project-thread-ls8p9r` (based on pipeline-rebuild);
  Ruby pulls them into pipeline-rebuild. First commit c91c77b: clip_to, run-name check, deviation comments.

### Update 2026-10-15 (detect.py; draft + walkthrough in `notes/drafts/`)

- **Pushed c675f5e** to `claude/project-thread-ls8p9r`: `imagery.ee_project` (no default,
  CHANGE_ME refused) and `panel_pipeline/provenance.py` (plan.py's helpers moved unchanged:
  `environment(cfg)`, `fingerprints(cfg)`, `sha256`, `repo_path`). provenance.json format unchanged.
- **detect.py** (Ruby types it from `drafts/detect_walkthrough.md`): one command = one frequency x
  one subregion, looping over periods. Output `<run>/detect/<freq>/period=<tag>/subregion=<n>/part_*.parquet`
  plus `_status.json` (Hive folders, so `pd.read_parquet(<run>/detect/monthly)` reads all, `_` files skipped).
- **Columns:** chip_id, confidence (float64 = AMW's exact value; float32 would flip `> 0.4` for a
  score of exactly float32(0.4)), vote_1..6 (float32; members in `_status.json`), n_obs, n_clear,
  empty_share (float32 chip means). Defaults proposed to Ruby 2026-10-15; no answer yet.
- **Reuse:** subclass of AMW's GEE_Data_Extractor (only `_build_composite` extended with two count
  bands from the same collections and cs_cdf test); AMW's get_tile_data, cut_chips, scoring lines.
  Tested: bit-identical to AMW's InferenceEngine.predict_on_tile_pixels on 3,174 chips (fake imagery).
- **No image cache** (~200-300 TB; AMW caches uint16). Resume from part files; AMW's 3 tries;
  failures with error text; session records (code, packages, host) written at start and end.
- **Not yet tested against real Earth Engine** (no EE login in Claude's container); Ruby's first test is
  `--limit 1` on 2024-07. Ruby's Earth Engine project (2026-10-15): `deforestation-and-mining`
  (told her to use the project ID if the console shows a different one).
- **Pushed a59a9b6** (Ruby asked for the project as its own config variable, not under `imagery:`):
  config.yaml now has `earth_engine: project: "deforestation-and-mining"` (committed value);
  config.py exposes it as `cfg.ee_project`; detect.py reads `cfg.ee_project` (drafts and walkthrough updated).
- **Pushed 2627fc2** (2026-10-19): plan.py keeps only polygon parts of each subregion ∩ clip_to; a clip touching a
  subregion only along its border (e.g. clip_to = Amazon_ACA_6.geojson) used to crash with DLTile InvalidShapeError.
  Pilot plan unchanged. Subregion 6 alone: 46,652 tiles, 24.7M chips, plan 18 min / 804 MB peak / chips.parquet 215 MB.
  Tiles per subregion: 1: 45,451, 2: 53,084, 3: 34,825, 4: 42,520, 5: 53,588, 6: 46,652; 274,932 unique.
  detect.py on fake imagery, 80 pilot tiles: 2.2 GB peak with 8 workers (1.3 GB with 2), ~3.4 s/tile on 4 cores.
- **detect.py is now in the repo** (Ruby typed it; 3d9c86e on pipeline-rebuild, identical to the draft except the
  old `cfg.imagery.ee_project` lines). The "Quarterly run and parallel layout" thread extended it on
  `claude/project-thread-jwtt0k` @ 85977db (merges pipeline-rebuild 30fdb3c and 2627fc2): `cfg.ee_project` fix,
  `--workers/--batch-tiles/--tries` overrides, `lock.py` per-folder `_lock.json`, signal handling, `status.py`.
  Base any further detect.py change on that branch; `notes/drafts/detect.py` is now out of date.

---

## 1. Can SAM2 be bolted on later? Yes, if we lock in five things now

SAM2 would be a separate script that reads the finished pipeline output, does
its own work, and writes **its own table** with the same keys `(chip_id, period)`.
"Adding columns" then means joining two tables, so the main panel is never
rewritten. For that to work later, the pipeline must provide now:

1. **A stable chip ID and a chip geometry table** (chip ID → its rectangle),
   written once. SAM2 needs the rectangles both to build its prompts and to count
   mask pixels per chip.
2. **The panel keyed by `(chip_id, period)`**, so any later table can be joined
   onto it.
3. **The prompt rule must be computable from panel columns.** SAM2's prompts are
   "chips passing the loose dual threshold" plus "Andes chips with mean ≥ 0.2".
   Both will be columns (`det_loose`, `in_andes`, `prob_mean`).
4. **A frozen config per run.** SAM2 must build its composites with exactly the
   same dates and cloud settings. It reads them from the run's saved config
   instead of retyping them.
5. **Nothing from the detector's imagery.** Their SAM2 already downloads its own
   256 px tiles (`sam2_mask.py:184`). So deferring SAM2 costs no extra Earth
   Engine traffic, and we don't need to keep detector imagery for it.

So yes: their design already keeps SAM2 separate, and our panel only has to keep
those five promises.

---

## 2. How their scoring actually ran

From `scripts/pipeline.py:151` and `core/README.md`:

- For each period the driver **prints 7 commands**: one per Amazon subregion (6)
  plus one for Andes. A person pasted them into `tmux` sessions (terminal
  windows that keep running after you log out).
- The README says they used **two VMs, with several of these processes running
  at the same time on each**. So it was not one VM per subregion, but processes
  spread over 2 machines.
- Each process scored *all* tiles of its subregion, for one period, from start to
  finish.
- **Only after every process had finished** did someone run concat, filter,
  postprocess and persist-detections. You are right about that.

The synchronisation was a human looking at terminals. That is the part we
replace.

---

## 3. The vocabulary (the part that feels like a black box)

| Word | Meaning | In our pipeline |
|---|---|---|
| **Node** | One computer in the cluster | Where a job runs |
| **Job** | A request to the scheduler: "run this script with X CPUs, Y GB memory, for at most Z hours" | One detect task, one assemble, one persist |
| **Scheduler** (usually **Slurm**) | The cluster's queue. It holds jobs until a node has room, then starts them | Replaces the person with `tmux` |
| **Job array** | One submission that means "run this same script N times, numbered 0…N-1" | All detect tasks of one period |
| **Array throttle** `%N` | "Of those, run at most N at the same time" | Our Earth Engine queue (§4) |
| **Dependency** | "Start job B only after job A finished OK" (`--dependency=afterok:A`) | Assemble waits for its period's detect tasks |
| **Process** | One running program with its own memory | One detect task = one Python process |
| **Thread** | A worker *inside* a process, sharing its memory | The "waiters" and the "chef" from round 2 |

How they fit together: Slurm puts **jobs** on **nodes**. Each job runs one
**process** (our script). Inside it, several **threads** download while one
scores. An **array** is many jobs of the same kind, and a **dependency** chains
one stage to the next.

---

## 4. My suggestion: four scripts plus a driver

Your idea (config, then inference, then concat + postprocess + persist) is close.
I'd split the last part in two, for a reason that matters for both speed and
storage:

- **postprocess needs space.** For one period, it needs every chip within 3 km,
  across subregion borders.
- **persist needs time.** For one chip, it needs every period.

Those are opposite ways of reading the data, and they can run at different
moments. Assembling period P can start as soon as P is scored, while other
periods are still running. Persistence has to wait for everything.

```
config.yaml ──► 1. plan.py            once per run
                   │  writes: frozen config, tile table, chip table, task table
                   ▼
                2. detect.py           thousands of jobs, in parallel
                   │  one job = one (period, subregion, shard of tiles)
                   │  writes: part files, done list, failed list
                   ▼   (per period, when all its jobs are done)
                3. assemble_period.py  one job per period
                   │  checks completeness → concat + dedupe → postprocess variables
                   │  writes: one panel file per period
                   ▼   (when all periods are done)
                4. persist.py          one job (or a few, split by area)
                      writes: final panel with persistence variables
   later:       5. sam2_*.py           reads 1–4, writes its own table

   submit.py:   turns the task table into Slurm arrays with throttles and
                dependencies, so nothing is typed by hand
```

Shared code (Earth Engine fetching, the model, chip cutting, dedupe, the
dual-threshold rule) lives in one library that all scripts import. Those are the
parts adapted from `core/`.

### Why each script is a separate program

- **Different resources.** detect needs many small CPU jobs, assemble a few
  larger-memory jobs, persist one long job. Slurm can only size a job if it is
  its own script.
- **Restart only what failed.** If assemble crashes on one period, you rerun
  that one job, not the scoring.
- **Easy to check.** Each step's output is a file you can open and inspect
  before the next step runs.

### Against splitting this much

- More files and more moving parts, and the driver must get the dependencies
  right.
- My answer: the driver is generated from the same task table, so it is
  mechanical. Every script also checks its inputs itself (§5), so a wrong order
  fails loudly instead of producing wrong data.

---

## 5. Routing: how every result lands in the right place

This is the core of your worry about running many periods and subregions at
once. The rule: **no output path is ever typed; each one is computed from the
task.**

1. `plan.py` writes a **task table**, with one row per
   `(period, subregion, shard)` and the exact list of tiles in that shard.
2. Slurm starts array job number 1234. The script reads row 1234 and from that
   *alone* knows the period, the dates, which tiles to score, and where to write.
3. Every task writes only inside its own folder, so **two tasks can never write
   the same file.** That's why any number of periods and subregions can run at
   the same time on any nodes.
4. Each result file also carries its identity *inside* it: period, subregion,
   tile key and the run's config fingerprint. If a file ever ended up in the
   wrong place, assemble would see that it doesn't match.
5. Files are written under a temporary name and renamed when complete. So a
   half-written file never looks finished.
6. Before concatenating, `assemble_period.py` compares **expected vs present by
   name**: every task done, every tile either scored or listed as failed. The old
   `publish` stage used the same idea (it checks names, not counts). Anything
   missing stops the step with a clear message.

Border tiles belong to two subregions, so two tasks score them. They land in two
different folders, and assemble removes the duplicate with their
max-confidence rule. That keeps their logic exactly.

The **config fingerprint** (a "hash", a short code computed from the config
file's contents) also names the run folder. If you change a setting, you get a
new run folder. Old and new outputs can then never mix, which fixes the old
problem that cloud settings weren't in the filenames.

### Folder layout

```
<RUN>/                                   e.g. run_2026-10-08_a1b2c3
  config.yaml                            frozen copy + git commit
  plan/tiles.parquet                     tile key, subregion(s), in_andes
  plan/chips.parquet                     chip_id, tile key, row, col, rectangle
  plan/tasks.parquet                     task_id, period, subregion, shard, tiles
  detect/<period>/<subregion>/shard_<n>/part_<k>.parquet
  detect/<period>/<subregion>/shard_<n>/failed.parquet
  periods/<period>.parquet               after assemble
  panel/                                 final, after persist
  logs/
```

### Sizing (estimates; the first pilot measures the real numbers)

- **Shard size:** around 500 tiles. That gives about 540 jobs per period and one
  array per period, which fits the usual Slurm cap on array size.
- **One detect job:** 1 process, about 8 download threads and 1 scoring thread,
  4–8 CPUs, 4–8 GB memory, **no GPU**. The CNN on 48 px chips is light; this is
  my inference, to be confirmed in the pilot. GPUs are only needed later, for
  SAM2.
- **The Earth Engine queue** is the throttle. The number of requests running at
  once = (jobs running at once) × (download threads per job). With `%N` on the
  array we keep that under your project's limit. Even on the partner tier there
  is a limit. When Earth Engine says "too many requests", the code waits and
  tries again; that part exists already.
- **Order:** the driver submits one array per period, then an assemble job per
  period that waits on that array, then persist, which waits on all assembles.
  Periods run side by side.
- **Restart:** resubmit the same array. Finished shards and tiles are skipped.

---

## 6. Storage: where things live, what we cache, when we clear

### Clusters usually have four kinds of storage

The names and limits differ per cluster, so I need yours to fill this in
(question 2 below).

| Kind | Typical properties | What we put there |
|---|---|---|
| **Node-local disk** (`$TMPDIR`) | Fastest; wiped when the job ends | The in-flight image cache of one job |
| **Scratch** | Large, fast, *not backed up*, often auto-deletes files untouched for N days, often a limit on the number of files | detect part files and period files *during* a run |
| **Project/work storage** | Backed up, smaller quota | Code, frozen configs, plan tables, the final panel, run logs |
| **Outside the cluster** (university archive or a cloud bucket) | Safe copy | A second copy of the final panel |

So, to answer you directly: **yes, the output is in a different place from the
cache.** The cache lives and dies with each job. Outputs are on scratch while the
run is going, and the final panel moves to backed-up storage.

### What we cache

- **Detector imagery: no long-term cache.** Keeping it all would be about
  200 TB (round2 §9). Each job keeps its tiles in memory or on its node-local
  disk until they are scored, then they are gone. Restarts don't need the
  imagery, because the part files record what's finished.
  - *Against:* re-scoring with a different model later means downloading
    everything again.
  - *Middle option:* keep a small fixed **sample** permanently, for example
    100 tiles × all periods ≈ 70 GB, for debugging and model comparisons.
- **SAM2 imagery:** later, and small (round2 §7).

### Too many files is a real risk

One file per tile would mean 270,000 tiles × 140 periods ≈ **38 million files**.
Cluster file systems limit the number of files, not just their size. So detect
writes one **part file per batch of tiles** (for example 50–100 tiles each,
which also limits what a crash can lose). Assemble then merges a period's parts
into one file and deletes them.

### When we clear

| What | Cleared when | By |
|---|---|---|
| Node-local image cache | End of each job | Automatic |
| detect part files of period P | After assemble P has passed its checks | assemble script |
| Period files | After the final panel has passed its checks (or kept) | persist script / you |
| Final panel | Never; also copied off the cluster | you |

### Size estimate

About 360 GB for the final panel before compression (round2 §9). Compressed
probably a third to half of that (inferred; the pilot measures it). At peak,
scratch holds part files plus period files. If parts are cleared as each period
is assembled, that is roughly one panel's worth.

**File format:** Parquet. It stores data by column and compressed, and Python,
R and Stata (through Arrow) can all read it. The final panel will be split into
files by area (groups of tiles), so each chip's full time series is in one place.
That's what persist needs and what most panel regressions read. We decide the
details in `persist.py`.

---

## 7. Questions (rephrased, all answerable without technical knowledge)

1. **Region:** `Amazon_ACA` (their current production area; I'd restore the
   files from git history) or `amazon_basin` (the only one on your branch)?
   I recommend `Amazon_ACA`: it is what their recent outputs used, and it
   contains all of the Andes area.
2. **Which cluster is it?** The name of the institution or cluster, or a link to
   its user documentation, is enough. I'll read it and fill in the scheduler and
   storage names myself.
3. **One cluster or several?** You wrote "on different clusters". If it's really
   several clusters that don't share a file system, results have to be copied
   between them, and the design changes. One cluster with many nodes is what I've
   assumed.
