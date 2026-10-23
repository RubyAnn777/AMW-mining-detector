# Workers, locks and status: how detect.py runs in parallel safely

Written 19 Oct 2026, **updated 23 Oct** for the streaming downloads and Ctrl-C saving of PR #1
(§1 Level 2, `--batch-tiles`, the signals table). It explains `panel_pipeline/detect.py`,
`lib/lock.py` and `status.py`. Everything described here was tested
with fake Earth Engine downloads: two copies at once, Ctrl-C, a scheduler stop, `kill -9`,
a lock left by another machine, a lock stolen mid-run, and failing tiles (§5). Real Earth
Engine was not part of those tests.

Contents:
1. The worker logic: three levels of parallel work
2. Run-time overrides: `--workers`, `--batch-tiles`, `--tries`
3. The lock file
4. status.py
5. What was tested
6. Your decisions (19 Oct)
7. Git: "divergent branches"

---

## 1. The worker logic: three levels of parallel work

Some vocabulary first:

- A **process** is one running program, e.g. one `python panel_pipeline/detect.py …` command.
  It has its own memory. Two processes share nothing except files on disk and Earth Engine.
- A **thread** is a line of work *inside* a process. All threads of a process share its memory,
  so they can hand each other data without files.
- Work is **I/O-bound** when it mostly waits on something outside the computer, here Earth
  Engine sending pixels. Work is **CPU-bound** when it keeps the processor busy, here the model
  scoring 529 chips.

Python has a rule called the **GIL** (global interpreter lock): only one thread runs Python code
at any moment. That sounds like threads are useless, but a thread that is *waiting* (on the
network, on disk) lets go of the GIL. So threads are ideal for downloads (mostly waiting) and
useless for heavy Python arithmetic. TensorFlow does its arithmetic in C++, outside the GIL, with
its own threads.

detect.py uses parallel work on three levels:

| Level | What runs in parallel | How many | Set by |
|---|---|---|---|
| 1. Tasks | whole `detect.py` processes, one per (frequency, period, subregion) folder | as many as you start; on the cluster, as many as the scheduler allows | you / the scheduler |
| 2. Download workers | threads inside one process, each waiting on one Earth Engine request | `--workers` (default: the run's `imagery.max_workers`) | `stream_tiles` (detect.py) |
| 3. Model threads | TensorFlow's own threads, scoring one tile's 529 chips | the CPU cores the process may use | TensorFlow |

### Level 2 in detail: what `stream_tiles` does

One process works through a whole period as one stream. There are no batches for the
downloads any more (until 20 Oct there were, see below):

```
 main thread                                     worker threads (here --workers 3)
 ───────────                                     ─────────────────────────────────
 hand out 6 downloads (2 × workers) ───────────► w1: tile A   w2: tile B   w3: tile C
                                                 (D, E, F wait in the queue)
 wait for whichever finishes first ◄──────────── w2: B done → w2 starts D
 score B (TensorFlow, all cores)                 w1, w3 still waiting on A, C
 hand out 1 more download (G joins the queue)
 wait for the next one ◄──────────────────────── w1: A done → w1 starts E
 score A …
 every 2 minutes, or every --batch-tiles tiles:
   write what is scored as part_0000N.parquet, update _status.json, one line in _log.txt
 … until the period is done, then merge all parts into part_00000.parquet
```

- The **main thread** does the scoring and all the writing. Workers only download. So only one
  thread ever writes, and a tile is never half-written by two threads.
- **Download and scoring overlap**: while the model scores tile B, Earth Engine is already
  preparing A, C and D. That's the whole point of the workers.
- **Downloads never pause.** Saving happens in between scoring two tiles; the workers keep
  downloading meanwhile. (Before PR #1, a batch of 500 had to finish completely before its part
  was written and the next batch started, so workers stood idle while the slowest downloads of
  each batch trickled in, and a stop lost the whole batch. That is AMW's structure,
  inference_engine.py:1167-1211.)
- **2 × workers in flight** (`itertools.islice(waiting, 2 * workers - len(running))`): enough
  that a worker never stands idle, and few enough that memory stays small. A tile is 18.6 MB
  since detect.py asks for 32-bit pixels (34.5 MB before), so 32 tiles in flight is about
  0.6 GB. AMW starts all 500 downloads of a batch at once (about 9 GB).
- **Failures**: if a download or scoring step raises an error, the tile is noted in `failed` with
  its error text (and in `_log.txt`), and the next `try` pass retries it (`--tries`, default 3).
  AMW's `get_tile_data` also retries short network hiccups by itself for up to 240 s
  (`@retry.Retry(timeout=240)` in core/gee.py), and a request that stays silent for 300 s is
  given up (`REQUEST_TIMEOUT_S`; added after a closed laptop lid froze a run).
- **The progress line** (screen and `_log.txt`, after every save) shows tiles per minute,
  download times, how busy the workers and the scoring are, Earth Engine retries, and the hours
  left. "workers busy" near 100% with "scoring busy" low means downloads are the limit.

### How to choose `--workers`

The speed of one process is set by the slower of the two pipes, downloading or scoring:

- **Downloads are the bottleneck** (the model waits for tiles): CPU use of the process is low
  (Activity Monitor on the Mac, `top` on the cluster: well under 100% per core it may use).
  More workers help, up to the point where Earth Engine answers "429 Too Many Requests"; those
  appear as failed tiles in `_status.json` and in `status.py`.
- **Scoring is the bottleneck** (tiles wait for the model): CPU near 100% of every core.
  More workers don't help, because there is nothing left to give them. More processes on more
  cores or machines do.

**Earth Engine sees the sum over all processes**: requests in flight = processes running × workers.
That number is what Earth Engine's limits apply to, so on the cluster the throttle is
"how many tasks at once" times `--workers`. The test with a few hundred tiles (Madre de Dios)
is where to find it: try `--workers 4`, `8`, `16` and compare the tiles per minute printed in
the log.

### How to choose `--batch-tiles` and `--flush-minutes`

Since PR #1 these only decide **how often a part file is written**, never the speed: a part is
saved when `--batch-tiles` tiles are scored or `--flush-minutes` (default 2) have passed,
whichever comes first. A crash or a time limit therefore loses at most about 2 minutes of work.

Many small parts are no problem any more: when a period is complete, detect.py merges them
into one `part_00000.parquet`. So a finished run has one data file per task folder.

Scores never depend on either setting: the same tile gets the same score in any part.

---

## 2. Run-time overrides: `--workers`, `--batch-tiles`, `--tries`

```bash
python panel_pipeline/detect.py --run <run> --frequency quarterly --subregion 2 \
    --from 2026Q1 --to 2026Q1 --workers 16 --batch-tiles 100 --tries 2
```

The run's `config.yaml` is frozen by plan.py, on purpose: a half-finished run must never change
how it scores. But these three settings change only *how fast* and *in what pieces* a period is
scored, never *what* a score is, and the right values differ between your MacBook and a cluster
node. So detect.py now takes them from the command line, and falls back to the frozen values.

How it works (`with_overrides` in detect.py):

- `Config` is a **frozen dataclass**: its fields can't be changed after it is built
  (`cfg.imagery.max_workers = 16` raises an error). That's what makes the frozen config
  trustworthy.
- `dataclasses.replace(cfg.imagery, max_workers=16)` builds a *new* `Imagery` with that one
  field changed and everything else copied. Then `replace(cfg, imagery=…)` builds a new `Config`
  around it. The original is untouched; detect.py just uses the copy.
- Every session in `_status.json` now records the values it actually used (`max_workers`,
  `batch_tiles`, `tries`, next to `limit`), so a period scored with 16 workers on the cluster
  says so.
- Settings that *do* change a score (imagery recipe, tile size, stride, model, thresholds) have
  no override. Changing them means a new run.
- Values below 1 are refused, like `--limit`.

---

## 3. The lock file

### The problem it solves

Two copies of detect.py writing the same folder at the same time is a **race condition**: the
result depends on which process happens to be faster at each moment. It can happen by accident,
for example when a task list is resubmitted while the first submission is still running, or when
you start a command in a second terminal and forget the first. Concretely:

1. Process A and process B both finish a batch at about the same time.
2. Both look at the folder, see `part_00003.parquet` as the highest number, and both decide to
   write `part_00004.parquet`.
3. B's file replaces A's. A's 500 tiles are gone, and A believes they are saved.

Before today, nothing stopped this. The duplicate check in `saved_tiles` would only notice some
of it, the next time a process started.

### The idea: one file that only one process can create

Before touching a period's folder, detect.py creates `_lock.json` in it. The trick is **how** it
creates it:

```python
fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
```

`O_CREAT | O_EXCL` asks the operating system "create this file, but only if it doesn't exist
yet", as **one indivisible step** (an *atomic* operation). If two processes ask at the same
instant, exactly one succeeds and the other gets `FileExistsError`. Writing it as two steps,
"check whether the file exists, then create it", would not work: both processes could check,
both see "no file", and both create it. That gap between checking and acting is the classic
cause of race conditions.

Mac file systems (APFS) and the usual cluster file systems (ext4, NFS version 3 and newer,
Lustre, GPFS) all honour `O_EXCL`.

### What is in the lock

```json
{
  "token": "fb54c883cdca445b97b9e664b75156ab",
  "host": "Mac.fritz.box",
  "pid": 50213,
  "job": null,
  "started_utc": "2026-10-19T08:02:48+00:00",
  "command": "panel_pipeline/detect.py --run … --frequency quarterly --subregion 2 --from 2026Q1 --to 2026Q1"
}
```

- `token`: a random name only this process knows (`uuid4`), used for the ownership check below.
- `host`, `pid`: which machine, and which process on it (the process ID).
- `job`: the scheduler's job ID when there is one (Slurm, PBS, LSF or SGE variables), else `null`.

The file starts with `_`, so Parquet readers (pandas, pyarrow, DuckDB, the notebook) skip it,
just like `_status.json`.

### The life of a lock

In detect.py, each period runs inside

```python
with FolderLock(folder) as lock:
    status = run_period(..., lock)
```

A **`with` block** (a *context manager*) guarantees that the cleanup code (`__exit__`) runs when
the block ends, however it ends: normally, by an error, or by a stop request. It's the same
mechanism as `with open(file) as f:` closing the file. In order:

1. **Take**: `__enter__` creates the file atomically (above). If it exists and its holder is
   alive, `Locked` is raised, and detect.py prints `SKIPPED: … is being written by <host>
   (process <pid>) …` and moves on to its next period. At the end it exits with an error that
   lists the skipped periods.
2. **Heartbeat**: a small background thread touches the file once a minute (`os.utime`, which
   sets its modification time to "now"). The file's age therefore says when its holder was
   last alive. It is a *daemon* thread, so it never keeps Python from exiting.
3. **Check before every write**: before each part file and each status update,
   `lock.check()` reads the lock and compares the token with its own. If the lock is gone or
   belongs to someone else, it stops with `LOCK ERROR … stopping without writing`. This is
   called a **fencing token**. It protects against the rare case where this process was frozen
   (a sleeping laptop, an overloaded node) long enough for another process to take over.
4. **Give back**: `__exit__` deletes the lock, but only if the token is still its own.

### Stop requests: signals

When something wants a program to stop, the operating system sends it a **signal**:

| Signal | Sent when | Before today | Now |
|---|---|---|---|
| SIGINT | you press Ctrl-C | a Python traceback (there was no lock yet) | saves what is scored, then a clean exit, lock given back; a second Ctrl-C stops at once |
| SIGTERM | a scheduler's time limit, `kill <pid>` | Python ended at once, no cleanup | saves what is scored, clean exit, lock given back |
| SIGHUP | the terminal window is closed | Python ended at once, no cleanup | clean exit, lock given back |
| SIGKILL | `kill -9`, out-of-memory killer, a lost node | — | can't be caught by any program: the lock stays behind (see stale locks) |

**Update 20 Oct (PR #1):** the first signal now only raises a flag. `run_period` checks it after
every tile, saves what is scored as a part, and then stops; nothing scored is lost. A second
Ctrl-C stops at once and loses only what was scored since the last save. The text below
describes the 19 Oct version, whose clean-exit logic is still there.

`stop_on_signal` in detect.py turns the first three into a normal Python exit
(`SystemExit("DETECT STOPPED by SIGTERM; saved parts are kept. Run the same command again to
resume.")`), so `with` blocks run their cleanup. To make that quick, `stream_tiles` (then `score_batch`) ends its
thread pool with `shutdown(wait=False, cancel_futures=True)`: downloads still queued are
dropped, and running ones are not waited for. That's safe because workers only download;
anything they fetch after the stop is never scored or written. A tile in the middle of a batch is
simply redone next time; a half-written part file starts with `_` and is deleted at the next
start (that was already the case).

**Update 19 Oct: a second Ctrl-C.** `shutdown(wait=False)` stopped detect.py from waiting
for the downloads, but Python itself still waited for them when it exited (it joins every pool
thread at exit, and one Earth Engine request can retry for up to 240 s). Pressing Ctrl-C again
during that wait ran `stop_on_signal` inside Python's own shutdown code, which printed the
"Exception ignored in threading … t.join() … SystemExit" traceback. Nothing was lost (the lock
was already given back and every part was complete), but it looked broken. Now the first signal
makes detect.py ignore further signals while it cleans up, then it prints the message and leaves
with `os._exit(1)`, without waiting for downloads it will never use. Tested: the old code
reproduces the traceback; the new one exits within a tenth of a second of Ctrl-C or SIGTERM.

### Stale locks: when the holder died without cleaning up

A lock is **stale** when its holder is gone:

- **On the same machine**, detect.py can ask the operating system directly:
  `os.kill(pid, 0)` sends no signal, it only answers whether that process exists. A dead process
  makes the lock stale at once. On your Mac, a `kill -9` is therefore recovered by simply
  running the command again.
- **From another machine** (a cluster node, the login node), only the heartbeat tells. A lock
  untouched for **15 minutes** (`STALE_AFTER_S`) is stale, because a living holder touches it
  every minute.

The next detect.py that finds a stale lock prints `taking over a stale lock …: <who held it>`,
deletes it, and takes its own. Nothing is lost: the part files are complete, and the resume logic
skips the tiles already saved.

### What you'll see, case by case

| Situation | What happens |
|---|---|
| Normal end | lock deleted, `_status.json` complete |
| A second copy on the same period | the second one prints `SKIPPED`, leaves the folder alone, exits with an error naming the skipped periods |
| Ctrl-C, closed terminal, time limit | what is scored is saved, `DETECT STOPPED …`, lock deleted, rerun resumes |
| `kill -9` / out of memory / node lost | lock stays; `status.py` shows `S` (stopped without cleaning up); rerun takes it over (same machine at once, elsewhere after 15 min) |
| Holder frozen > 15 min, then wakes up | someone may take over; the frozen one stops at its next write with `LOCK ERROR` |

**Don't delete `_lock.json` by hand** unless `status.py` shows that folder as `S`, and even then a
rerun does it for you.

---

## 4. status.py

```bash
python panel_pipeline/status.py --run tests_output/test_region_small_2026-10-19
python panel_pipeline/status.py --run … --frequency quarterly
python panel_pipeline/status.py --run … --todo > tasks.txt
```

It reads the run's frozen config, `plan/tiles.csv`, and each folder's `_status.json` and
`_lock.json`. It **writes nothing**, so it's safe to run any time, also while detect.py runs.
It needs no Earth Engine login, TensorFlow or geopandas, so it starts instantly, also on a
cluster's login node.

Example (fake data, two subregions with tiles, 2 and 3; each cell is one mark per subregion):

```
run         tests_output/demo_2026-10-19
subregions  2, 3 (those with tiles; one mark each, in this order)

monthly: 210 tasks: 5 done, 1 running, 1 failed, 3 partly done, 200 not started
      Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
2018  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
…
2026  #.  #.  #.  ~.  #.  #.  ~.  !.  ~R

#  done   R  running   !  has failed tiles   ~  partly done   S  stopped without cleaning up   .  not started

running now:
  monthly 2026-09 subregion 3: node042 (process 31877, job 991234_57), since 2026-10-19T08:01:12+00:00, last heartbeat 0 s ago

look at these:
  monthly 2026-04 subregion 2: 8/12 tiles saved
  monthly 2026-08 subregion 2: 2 tile(s) failed, e.g. 552:12:10.0:20:11:104: RuntimeError: fake HTTP 429 Too Many Requests
```

Reading the 2026 line: in January, `#.` means subregion 2 is done and subregion 3 is not
started; in September, `~R` means subregion 2 is partly done and subregion 3 is running.
For the full run each cell has six marks, one per subregion.

How it decides a task's mark, in this order:

1. a lock with a living holder → `R`; a stale lock → `S`
2. no `_status.json` → `.`
3. `complete: true` → `#`
4. failed tiles listed → `!` (with the first error)
5. otherwise → `~` (with tiles saved / planned, and `--limit` if the last run used one)

To keep this current, **detect.py now rewrites `_status.json` after every saved part**, from the
counts it already holds in memory. It no longer re-reads the part files for that, so the update
costs a few milliseconds. Before today the record was written only at the start and end of a
session, so a stopped run looked like "0 tiles saved".

### `--todo`: the task list

`--todo` prints one detect.py command per task that is neither done nor running:

```
python panel_pipeline/detect.py --run … --frequency monthly --subregion 2 --from 2018-01 --to 2018-01
python panel_pipeline/detect.py --run … --frequency monthly --subregion 3 --from 2018-01 --to 2018-01
python panel_pipeline/detect.py --run … --frequency monthly --subregion 2 --from 2018-02 --to 2018-02
…
```

- **Period first**, as you decided: all subregions of 2018-01, then all of 2018-02, and so on.
  Monthly comes before quarterly, as decided on 12 Oct. `--frequency quarterly` gives only the
  quarters.
- This file *is* the task list for any scheduler: line *i* is task *i*. How it gets started on
  the cluster is up to the setup you choose there.
- **Resubmitting is always safe**: running `--todo` again lists only what is still missing, and
  even a task listed twice can't do harm, because the lock makes the second copy skip.

---

## 5. What was tested

On fake Earth Engine downloads (a stand-in for AMW's `gee.py`, a fake model, the real
`detect.py`, `lock.py`, `status.py`, `config.py` and `periods.py`), with your run's frozen config
and model fingerprint:

| Test | Result |
|---|---|
| normal run with `--workers 3 --batch-tiles 5` | 3 part files per period, all chips, session records 3 / 5 / 3 |
| `--workers 0` | refused: `--workers must be at least 1` |
| rerun of a finished period | done in 0.7 s, no download |
| second copy started on a running period | `SKIPPED … is being written by …`, no duplicate chips, first copy finished normally |
| `status.py` during a run | `R` and the holder in "running now" |
| SIGTERM mid-run | stopped in ~0.5 s, lock deleted, status "8/12 tiles saved", rerun completed |
| Ctrl-C mid-run | same as SIGTERM |
| `kill -9` mid-run | lock left, `status.py` shows `S`, rerun took it over at once (same machine), completed, no duplicate chips |
| lock from another machine, fresh | skipped |
| same lock, heartbeat 20 min old | taken over, completed |
| lock replaced mid-run by another token | `LOCK ERROR`, stopped before writing, the other lock left alone |
| two failing tiles, `--tries 2` | retried once, listed as failed with their error, `!` in status |
| heartbeat | an hour-old lock was refreshed within one heartbeat |
| `status.py` on your real test run | 3 monthly done, quarters not started |

---

## 6. Your decisions (19 Oct)

| Decision | Choice |
|---|---|
| Task unit | one task per (frequency, period, subregion) |
| Task order | period first: all subregions of a period, then the next period; monthly before quarterly |
| Naming | `tests_output/<area>_<YYYY-MM-DD>`, one committed config per run in `configs/` |
| Run-time overrides | built (`--workers`, `--batch-tiles`, `--tries`) |
| Lock file, status.py | built |
| Cluster setup | different from the sketch; to be designed when you move there |

---

## 7. Git: "divergent branches"

`fatal: Need to specify how to reconcile divergent branches` means both sides got new commits
since they last met. Your `pipeline-rebuild` had your 4 commits from today, and my branch had the
notebook commit. Git can then either **merge** (keep both histories and add a commit joining
them) or **rebase** (replay your commits on top of mine, which rewrites them). Recent Git
versions refuse to guess.

To make your pull simple, I merged your `pipeline-rebuild` into my branch. My branch now contains
all your commits plus mine, so for you the pull is a **fast-forward**: your branch pointer just
moves forward, with no merge needed. If this happens again with your own new commits in between,

```bash
git config pull.rebase false     # once: "when both moved on, merge"
```

makes `git pull` merge on its own. If it then opens an editor for the merge message, save and
close it. In vim that's `:wq` and Enter.
