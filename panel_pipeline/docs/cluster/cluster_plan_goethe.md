# Running the panel on Goethe: storage, queue and parallel plan

> **Revision 23 Oct 2026.** This block replaces §3.2–§3.7, §4 and parts of §2 below,
> which were written before the live-run measurements of 20–22 Oct and before the
> "nothing is deleted" decision (Rmd D11). The sections below are kept for their
> reasoning; where they disagree with this block, this block wins. The how-to is now
> `panel_pipeline/README.md`, Part B.

## R1. What changed since this plan was written

**Code** (`detect.py` on `claude/project-thread-v19nvg`, PR #1, which Ruby has pulled locally):
- Downloads never pause between batches; a part is saved every **2 minutes**
  (`--flush-minutes`) or every `--batch-tiles` tiles, whichever is first. A crash or a
  time limit now loses at most ~2 minutes, not a 500-tile batch.
- When a task completes, its parts are **merged into one `part_00000.parquet`**. So a
  finished run has 840 data files, not ~77,000. The "too many files" worry is gone.
- Each task folder has a `_log.txt`; Ctrl-C / SIGTERM save and exit cleanly; Earth
  Engine requests time out after 300 s of silence; pixels are requested as 32-bit, so a
  tile is 18.6 MB instead of 34.5 MB; the macOS-only TensorFlow setting is off by default,
  so a Mac scores the same way as Linux (`--mac-eager` restores it).
- Lock, overrides and `status.py` (19 Oct) are in, and `status.py --todo` prints the
  period-first task list the cluster runs from.

**Decisions** (`panel_pipeline/docs/open_design_decisions.Rmd`): the panel is complete and
additive; **nothing is deleted** (D11), so detect's parts stay next to the assembled
layers; one run at a time (P4); DuckDB open (D10).

**Measured** (live run on Ruby's Mac, subregion 3, 2026Q3, 34,825 tiles):
- Scoring: ~1.85 s per tile compiled on the M5; "scoring busy" 15–37%.
- The Mac's Wi-Fi was the limit: 20–23 tiles/min at 18.6 MB per tile with 16–22 workers,
  0 retries (so 22 concurrent requests are within the project's limit).
- Earth Engine's own work: **~130 EECU-seconds per tile-quarter** (56–213), ~35 s per
  request on its own. Restricting the recipe to the tile saved nothing reliable.
  A monthly composite is cheaper (fewer images) but not measured yet.

## R2. The real ceiling is Earth Engine, not the cluster

Google's documented limits for one project, whoever sends the requests
([quotas](https://developers.google.com/earth-engine/guides/usage),
[noncommercial tiers](https://developers.google.com/earth-engine/guides/noncommercial_tiers)):

| Limit | Value | Over it |
|---|---|---|
| Concurrent requests (high-volume endpoint) | **40 per project**, default; an adjustable quota | "429 Too Many Requests"; the client retries, detect.py counts it as *retries* |
| Compute, Partner tier | **100,000 EECU-hours per month** | not a hard cut-off: "restricted mode" (slower) until the 1st |

Arithmetic (*estimate* from the measurements above; the first cluster test must confirm):

- Throughput at 40 in flight, 35–60 s per request: **40–70 tiles/min ≈ 60,000–100,000
  tile-periods per day.**
- EECU at 130 s per tile-quarter: 100,000 h/month ≈ **92,000 tile-quarters per day.**
  The two limits coincide at roughly 100k a day.
- Scoring 100k tiles a day at 1.85 s per tile needs ~2 cores; with 4-core TensorFlow it
  is ~12 cores. **One Goethe node keeps up with everything Earth Engine will give us.**

| Scope | tile-periods | days at ~90k/day |
|---|---|---|
| full plan: 274,932 tiles × (105 months + 35 quarters) | 38.5 M | **~430 (about 14 months)** |
| quarterly only, 2018–2026 | 9.6 M | ~105 |
| monthly only | 28.9 M | ~320 (less if a month is much cheaper than a quarter in EECU) |
| one subregion (~45k tiles), all 140 periods | 6.4 M | ~70 |

**Consequences for the plan:**
1. §3.2 Option B's "20 worker nodes" is wrong by a factor of ~20. The right shape is
   **one job on one node** running 2 tasks side by side with ~20 workers each (= 40 in
   flight), resubmitted when it ends. That is `cluster/detect_tasks.sbatch`; no
   `worker.py` is needed for the MVP. A second node is only useful if Google raises the
   concurrency quota.
2. The number of nodes is not a decision any more; **the scope of periods is.** This is
   Ruby's call (R5).
3. The levers that could raise the ceiling, in order of promise: (a) ask Google to raise
   the project's concurrent-request quota (Cloud console → IAM & Admin → Quotas →
   Earth Engine; the EECU quota then binds at ~90k tile-quarters a day); (b) measure the
   monthly EECU cost: if a month is ~40 EECU-s, monthly scoring is EECU-cheap and only
   concurrency-bound; (c) fewer periods or a smaller region; (d) the Contributor→Partner
   application if the project is not Partner yet (R5).
4. Goethe's core-hour budget is no longer a worry: ~100k tile-periods a day on one node
   is ~1,000 core-hours a day at most.

## R3. Storage, revised

- Detect: 840 merged files, ~0.8 TB for the full plan (22 KB per tile-period).
- Nothing deleted: assembled layers add about the same again, plus added layers later.
  Plan for **~2–2.5 TB on `/work`** for the full plan, against the group's 5 TB. Check
  `quota` and what colleagues use before the full run.
- Backups as in §2: small records to `/home` after each session, finished periods and
  panels to `/arc` with `rsync` from a login node. The archive copy replaces the earlier
  "delete parts after assembly".
- At ~100k tile-periods a day, detect writes ~2 GB a day: the file system is never the limit.

## R4. Crashes, revised

| What breaks | Lost | Restart |
|---|---|---|
| tile fails (429 / 500 / 300 s timeout) | nothing; retried in the next try | automatic, then the task is in the next `--todo` |
| task crashes or node dies | ≤ 2 min of scoring; lock stale after 15 min | resubmit |
| job reaches 21 days, or `scancel` | nothing (SIGTERM saves) | resubmit |
| Earth Engine over quota | nothing; everything slows | wait for the 1st, or lower the load |

## R5. Open points for Ruby (23 Oct)

1. **Scope:** which periods and region to score first, given ~90k tile-periods a day.
2. Is the Cloud project on the **Partner** tier already? (Cloud console → Earth Engine →
   Manage tier.) Partner needs a separate application that "can take several weeks".
3. Whether to ask Google for a higher concurrent-request quota.
4. Group name on Goethe, and whether the group has `/arc` space.

---

**Status 19 Oct.** Built from the Goethe-NHR wiki page
(<https://wiki.csc.uni-frankfurt.de/wiki/doku.php?id=public:usage:goethe>) and your
answer that compute nodes reach Earth Engine. This replaces sections 4, 6 and 7 of
`cluster_migration.md`; its sections 2, 3 and 5 still hold. Numbers marked *estimate*
come from the Mac measurements (`quarterly_and_parallel_runs.md` §4) and must be
re-measured on Goethe (Part 6).

---

## 1. What Goethe tells us, and why it changes the plan

| Fact from the wiki | Consequence for us |
|---|---|
| `general1`: 412 Skylake nodes, **40 cores** (80 with hyper-threading), **192 GB**; AlmaLinux 9 x86_64 | Plenty. TensorFlow's normal Linux wheels run as is. |
| **`general1`/`general2` give you whole nodes, exclusively** | **The biggest change.** A job is never "4 cores", it is always 40. One detect.py at 4 cores would leave 36 idle and still count as a full node. So one node must run **several detect tasks at once**. |
| Per user: **40 running jobs, 50 submitted**, array ≤ 1001, **21 days** max wall time | We can't just submit 840 tasks. In Slurm, every element of a job array counts as one job against "50 submitted" (to confirm on Goethe, Part 6). |
| `test` partition: 2 h, 2 jobs, 2 nodes | Ideal for the pilot and the speed measurements. |
| `/home/<group>/<user>`: 30 GB, slow NFS, **backed up nightly** | Code, the key, small records. |
| `/work/<group>/<user>`: 5 TB default (group quota), fast PanFS, **no backup**, "delete data older than ~30 days" | The run folder lives here. |
| `/scratch`: discontinued Feb 2026 | Ignore. Notes saying "scratch" now mean `/work`. |
| `/local/$SLURM_JOB_ID`: up to 1.4 TB per node, slow, wiped at job end | We don't need it (no image cache in the production run). |

**Image cache, in one line:** the production run caches no imagery (handoff §4: ~200–300 TB,
and AMW's cache stores rounded pixels). A small *test* cache for development on your Mac is a
separate question, answered in the "Quarterly run and parallel layout" thread; if one is kept
on Goethe later, it goes in `/work` and counts against the same 5 TB.
| `/arc01`, `/arc02`: archive, ≤ 10 TB per group, **requested by your group leader**, reachable only from login nodes with `rsync`, no backup | The only big place that outlives `/work`. For the final panels. |
| Jobs that exceed their memory are killed | Set memory per job explicitly; measure it. |
| Login nodes: no heavy work | Install, git, submit, small tests only. |

---

## 2. Storage: where everything lives

```
/home/<group>/<user>/                         30 GB, backed up nightly, slow
├── AMW-mining-detector/                      git clone (code, model, boundaries: ~30 MB)
├── .secrets/ee-amw-panel.json                service-account key (chmod 600)
└── runs_records/<run>/                       small copy of what can't be rebuilt:
                                              run/config.yaml, run/provenance.json,
                                              plan/tiles.csv, logs/, every _status.json

/work/<group>/<user>/                         5 TB, fast, NOT backed up
├── venvs/amw-detect/                         the Python environment (rebuildable from the lock)
├── cache/uv/  and  python/                   uv's download cache and its Python 3.11
└── runs/<run>/                               the run folder (plan.py's output_root)
    ├── run/  plan/
    ├── detect/<frequency>/period=<tag>/subregion=<n>/part_*.parquet  _status.json  _lock
    ├── assemble/<frequency>/period=<tag>.parquet
    ├── logs/
    └── panel/

/arc0x/archive/<group>/amw_panel/<run>/       archive: assembled periods + final panels
```

**Why the venv goes on `/work`, not `/home`.** A TensorFlow venv is ~2 GB and tens of
thousands of small files. On slow NFS, 20 nodes all importing TensorFlow at once is slow,
and the 30 GB home quota is better kept for code. The venv needs no backup: the lock file
in git rebuilds it in a few minutes. *Against:* if `/work` is wiped, you rebuild it once.

**Why not write part files to `/local` first.** The wiki suggests computing on `/local`
and `rsync`-ing to `/work` at the end of a job. For us that is worse: a job killed at
day 3 would lose three days of results instead of one batch. Part files go straight to
`/work`, written temp-then-rename as today. `/local` stays unused.

**Size check (*estimate*):** detect output ≈ 0.6 TB monthly + 0.2 TB quarterly. At peak,
`/work` holds part files of unassembled periods + assembled periods + the panel, roughly
2–2.5 TB. That fits in 5 TB, but the group quota is shared with your colleagues, so
check `quota` before the full run. Part files are deleted only after their period is
assembled and checked (unchanged from the handoff).

**What survives what:**

| Lost | We still have |
|---|---|
| a node | everything; rerun its tasks |
| `/work` (disk failure, cleanup) | code, config, provenance, plan tiles, statuses, logs in `/home`; assembled periods already copied to `/arc`. Re-plan and re-detect only the rest |
| `/home` | nightly backup |
| the cluster | the panels copied off Goethe (Part 2, last step) |

**The backups we need to do ourselves:**
1. After each period is assembled: `rsync` it to `/arc` (from a login node, because `/arc`
   isn't mounted on compute nodes). A small script, run by hand once a day, is enough.
2. `runs_records/` in `/home`: `rsync` the small files after each session (seconds).
3. The final panels: `/arc` plus one copy off the cluster (university storage or your
   group's drive).

**Action for you:** ask your group leader whether the group already has `/arc` space
(`df -h /arc0{1,2}/archive/<group>` on a login node shows it). Without it, the only
backed-up place is home (30 GB), which is too small for the panels.

**About the 30-day rule on `/work`.** The wiki phrases it as "delete when no longer
needed, e.g. older than 30 days", which reads as a request, not an automatic purge.
Assembling and archiving each period soon after it finishes keeps us safe either way.

---

## 3. Queue and parallelisation

### 3.1 Three layers of parallelism, with Goethe's numbers

1. **Nodes** (Slurm jobs): up to 40 at once for you.
2. **Tasks per node** (processes): each runs one `detect.py --frequency f --subregion n
   --from P --to P`, with its own copy of the model. Separate processes, so a crash in
   one doesn't touch the others.
3. **Download threads per task** (`max_workers`): Earth Engine requests in flight.

Earth Engine load = nodes × tasks per node × `max_workers`.
CPU use per node = tasks per node × TensorFlow threads per task (must be ≤ 40).

### 3.2 The options for feeding tasks to nodes

**Option A: job-array waves.** One array element = one node = one period (its 6
subregion tasks side by side). 140 periods → 140 elements, but only 50 may be submitted,
so you submit 1–50, then 51–100 when those finish, then the rest.
- *For:* plain Slurm, nothing to write beyond the job script.
- *Against:* you babysit 3+ submissions per frequency; subregions differ in size
  (34k–53k tiles), so a node idles ~30% of the time while it waits for its biggest
  subregion; a period that fails needs its own resubmission.

**Option B: worker pool (my recommendation).** Submit a small array of long-running
"worker" jobs, e.g. 20 nodes for up to 21 days. Each worker runs a loop on its node:
take the next unfinished task from the run's task list, run it, take the next, until
the list is empty. It keeps, say, 8 tasks running at once on its 40 cores.
- *For:* one submission for the whole run (20 jobs ≪ 50); no idle cores while tasks are
  left; the task list is ordered **period first**, so periods finish in time order and
  can be assembled and archived as they come; restart = submit the workers again, and
  they skip finished tasks; the number of nodes is the Earth Engine throttle, and you can
  add or cancel workers while the run is going.
- *Against:* ~60 lines of new code (`worker.py`); two workers must never take the same
  task, which needs the lock file (the other thread is building it, see 3.4); a bug in the
  worker loop affects every node.

**Option C: one big job per frequency**, all tasks through GNU `parallel` on many nodes.
Fragile (one failure kills everything) and harder to restart. Not recommended.

**Why B fits our design.** Every task is already idempotent (a rerun skips saved tiles)
and writes only its own folder. A worker pool is the standard way to run many such tasks
under a "few jobs, long wall time" policy like Goethe's. It keeps the unit of work
exactly as you decided it: one task per (frequency, period, subregion).

### 3.3 What the worker looks like (sketch, to be written together)

```
worker.py --run <run> --frequency monthly --slots 8 --threads 5 --workers 8

loop:
    tasks = read <run>/tasks_monthly.txt          (period first: 2018-01 s1..s6, 2018-02 ...)
    while fewer than --slots tasks are running:
        pick the next task that is not complete and not locked
        start: python detect.py --run ... --frequency ... --subregion n --from P --to P
                                --workers 8 --threads 5      (run-time overrides)
               output → <run>/logs/<frequency>_<P>_s<n>.log
    when a task ends: record its exit code; a failed task can be picked again
                      up to 2 more times, then it is skipped and left for a human
    stop when nothing is left to start and nothing is running
```

The Slurm job script around it is about 10 lines:

```bash
#!/bin/bash
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --time=21-00:00:00
#SBATCH --mem=0                       # all of the node's memory (it's ours anyway)
#SBATCH --array=1-20                  # 20 worker nodes = the Earth Engine throttle
#SBATCH --output=<run>/logs/worker_%A_%a.out
module purge
source /work/<group>/<user>/venvs/amw-detect/bin/activate
cd ~/AMW-mining-detector
python panel_pipeline/worker.py --run /work/<group>/<user>/runs/<run> --frequency monthly --slots 8 --threads 5
```

### 3.4 Locks across nodes (input for the "Quarterly run and parallel layout" thread)

The lock from the "Quarterly run and parallel layout" thread is now on
`claude/project-thread-jwtt0k` (explained in `../code/workers_locks_and_status.md`).
It already works across nodes: `_lock.json` records host, process ID and Slurm job ID; a
background thread refreshes its heartbeat every minute; a lock from another machine
counts as stale after 15 minutes without a heartbeat; SIGTERM (Slurm's time limit) gives
the lock back cleanly. `O_CREAT | O_EXCL` creation is atomic on PanFS (`/work`), so it
holds on Goethe as designed. Nothing to add; the worker only has to treat "skipped:
locked by someone else" as "pick another task", not as a failure.

### 3.5 TensorFlow threads

On a 40-core node, TensorFlow inside each task would start 40 threads by default. With
8 tasks that's 320 threads fighting over 40 cores ("oversubscription"), which is slower
than fewer threads. So detect.py gets a `--threads` override that sets
`tf.config.threading.set_intra_op_parallelism_threads(n)` and
`set_inter_op_parallelism_threads(1)` before the model loads, plus
`OMP_NUM_THREADS=n`. Rule: slots × threads = 40. We use the 40 physical cores and ignore
hyper-threading at first; the benchmark (Part 6) tries 80 too.

### 3.6 Earth Engine throttle

20 nodes × 8 tasks × 8 downloads = **1,280 requests in flight**. That's a guess at what
the partner tier tolerates. The real number comes from the benchmark: if `_status.json`
shows HTTP 429 ("too many requests"), lower `--workers` or the number of worker nodes
(`scancel` some array elements; the others carry on). No code change either way.

### 3.7 Rough timeline (*estimate*)

~100,000 core-hours in total (2.3 s per tile on 4 cores, ~38.7 M tile-periods).
With 20 nodes × 40 cores that's ~125 hours, about **5–6 days** if Earth Engine keeps
up, longer if it throttles. Goethe-NHR runs on project allocations, so check that
your project's core-hour budget covers ~100k plus a margin for re-runs.

---

## 4. How often we write, and crashes

**Writes (unchanged).** One part file per 500-tile batch, about every 20 minutes per
task, straight to `/work`. Plus `_status.json` after each batch, the lock heartbeat, and a
log line per batch. A crash loses at most the current batch of each running task.

**Crashes on Goethe:**

| What breaks | What happens | Restart |
|---|---|---|
| tile fails (429/500/timeout) | retried `tries`=3 passes, then marked failed | the worker re-runs the task later; only failed tiles are fetched |
| a task crashes (bug, out of memory) | worker logs the exit code, continues with other tasks | up to 3 attempts per task, then flagged in `status.py` |
| a node dies | its tasks stop mid-batch; their locks stay | after 15 min without heartbeat the lock is stale, another worker takes the task |
| job reaches 21 days | Slurm sends SIGTERM, kills ~30 s later | submit the workers again |
| `/work` quota full | writing a part fails, task fails loudly | free space, workers retry |
| key revoked / login fails | every tile fails fast | fix the key, resubmit |
| maintenance downtime | all jobs end | resubmit |

So restart is always the same command: submit the workers again.

---

## 5. Software setup on Goethe (once)

```bash
ssh <user>@goethe.hhlr-gu.de
# uv: one program in ~/.local/bin, no admin rights
curl -LsSf https://astral.sh/uv/install.sh | sh
# keep uv's big folders off /home
echo 'export UV_CACHE_DIR=/work/<group>/<user>/cache/uv' >> ~/.bashrc
echo 'export UV_PYTHON_INSTALL_DIR=/work/<group>/<user>/python' >> ~/.bashrc
source ~/.bashrc
uv python install 3.11                         # its own Python; no module needed
git clone git@github.com:RubyAnn777/AMW-mining-detector.git ~/AMW-mining-detector
cd ~/AMW-mining-detector && git checkout pipeline-rebuild
uv venv --python 3.11 /work/<group>/<user>/venvs/amw-detect
source /work/<group>/<user>/venvs/amw-detect/bin/activate
uv pip sync requirements/detect-linux.lock     # the lock we still have to make (cluster_migration.md §2.3)
```

`~/.bashrc` is a file of commands your shell runs at every login. The two `export` lines
make the settings permanent. Cloning needs an SSH key from Goethe added to your GitHub
account (`ssh-keygen -t ed25519`, then paste `~/.ssh/id_ed25519.pub` under GitHub →
Settings → SSH keys).

Why uv's own Python rather than `module load python`: the wiki doesn't document a Python
module, and uv's build is the same 3.11 you use on the Mac. If Goethe's `module avail`
shows a 3.11, either works.

---

## 6. The order to get there (each step proves the next)

1. **Setup** (Part 5) and the **service account** (`service_account_guide.md`). Check:
   the login test passes inside a compute job.
2. **Confirm the limits** on a login node: `sacctmgr show qos format=name,maxsubmitpu,maxjobspu`
   (whether array elements count against 50), `quota`, `df -h /arc0{1,2}/archive/<group>`,
   your project's core-hour budget.
3. **Pilot, `test` partition** (1 node, < 2 h): La Pampa, one month, one task. Compare
   scores with your Mac's run of the same month (they should match to ~6 decimals;
   this also settles the oneDNN question in `cluster_migration.md` §5.7).
4. **Benchmark, `test` partition**: Madre de Dios (748 tiles), one month, run 4, 8 and 10
   tasks side by side (threads = 40 / tasks) and `max_workers` 8 vs 16. Read tiles per
   minute, memory peak (`sacct -j <id> --format=MaxRSS`), and 429 counts. This sets
   slots, threads, workers and the number of nodes.
5. **One full period** of the whole ACA (6 tasks, one node, `general1`), then assemble it.
6. **Full run** with the worker pool, monthly, then quarterly.

Code to write on the way (you type, I explain): the run-time overrides and lock are done
in the other thread; still to write are `--threads` in detect.py, the
`earth_engine: credentials` setting, the Linux lock file, `tasks.py`, `worker.py`, the
job script, and an `archive.sh` for the `/arc` copies.

## 7. Open points for you

1. Option B (worker pool) vs A (array waves). I've assumed B.
2. Does your group have `/arc` space, or who is your group leader to ask?
3. Your project's core-hour allocation on Goethe.
4. Your Goethe group name, for the paths above.

## 8. Open design choices (undecided; nothing switched)

### 8.1 DuckDB for assembling and querying the panel (moved here from "Panel assembly scripts", 21 Oct)

**Current choice: pandas + pyarrow.** Ruby is fine with these for now. DuckDB stays an
open option, to revisit when the panel is assembled on Goethe.

- **What it is:** a database engine that runs inside Python (`pip install duckdb`, no
  server). It runs SQL directly on Parquet folders, including our Hive layout
  (`detect/monthly/period=*/subregion=*/`), and streams through the data instead of
  loading it into memory.
- **Why it might matter on Goethe:** one whole period is ~18 GB in pandas
  (`quarterly_and_parallel_runs.md` §4), and the full monthly panel is ~15 billion rows.
  pandas needs everything it works on in memory; DuckDB can concatenate, deduplicate and
  join periods larger than the node's 192 GB by spilling to disk. pyarrow's `dataset`
  API can also stream, but with fewer operations (e.g. no window functions for
  persistence across periods).
- **For DuckDB:** less memory, often much faster on group-by/join; reads the same Parquet
  files, so it can be added for one step without changing what's on disk; R and Stata
  users can query the same files.
- **Against:** a second way of writing logic (SQL next to pandas), which you'd have to learn
  and which our `# AMW:` comment markers don't map to as directly; one more package in the
  lock file; results must be checked against the pandas version once.
- **When to decide:** when assemble or persist first runs on a full ACA period and we see
  its memory use. If pandas fits in a node's memory per period, there's no need.
