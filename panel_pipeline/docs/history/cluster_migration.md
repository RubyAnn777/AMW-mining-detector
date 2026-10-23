> **History, not maintained.** 19 Oct, the first map of the move. Superseded by `../cluster/cluster_plan_goethe.md` (revision R1–R5 of 23 Oct) and Part B of `../../README.md`.

# Moving the pipeline to the cluster: the map

> **Update 19 Oct:** the cluster is Goethe-NHR, and its compute nodes reach Earth Engine.
> Sections 4, 6 and 7 are replaced by `cluster_plan_goethe.md`; the login is in
> `service_account_guide.md`.
>
> **Update 23 Oct:** §2 is done (`requirements/panel.in`, `panel-linux.txt`, `panel-macos.txt`
> on `claude/project-thread-xsq2cx`). §5: items 2 and 3 are handled by environment
> variables in `cluster/goethe_env.sh` (no code change); 4, 5 and 6 are done in detect.py
> and lock.py; 7 is still to measure. The how-to is `panel_pipeline/README.md`.

**Status 19 Oct.** Written before the cluster documentation arrived, so every
cluster-specific name (paths, partitions, limits) is still a blank. Section 7 lists
exactly what I need from the documentation. Code references are to
`claude/project-thread-jwtt0k` and the `notes/drafts/detect.py` draft.

The move touches seven topics. They are listed in the order we should settle them,
because each one depends on the ones above it.

---

## 1. What a cluster is, and what changes compared to your MacBook

A cluster is many computers ("nodes") sharing one file system, with a **scheduler**
(usually Slurm) deciding who runs where.

- **Login node**: where you arrive by `ssh`. You edit, run `git`, install packages and
  submit jobs here. You do **not** run heavy work here; admins kill long processes on
  login nodes.
- **Compute nodes**: where jobs run. You never log in to them directly. You ask the
  scheduler for "4 CPUs, 16 GB, 12 hours", and it starts your command on a free node
  when one is available.
- **Shared file system**: every node sees the same folders (`/home`, `/scratch`, …),
  so a file written by a job on node 17 is visible from the login node.

What this means for us:

| On the MacBook | On the cluster |
|---|---|
| `python detect.py ...` in a terminal | the same command, but inside a job (`sbatch` or an interactive `srun`) |
| closing the laptop stops the run | a job keeps running after you log out; it stops at its **wall-time limit** |
| all cores are yours | you get only the cores you asked for, and must tell TensorFlow so (§5.3) |
| internet everywhere | compute nodes may have **no internet**. This is the single biggest unknown (§4) |
| Apple silicon (ARM), macOS | almost certainly x86_64 Linux. Different packages (§2) |

**"Run by hand" (handoff §2.7) still works on a cluster**, through an *interactive job*:
`srun --cpus-per-task=4 --mem=16G --time=4:00:00 --pty bash` gives you a shell on a
compute node, and you type the same commands as on the Mac. Put it inside `tmux` on the
login node first, so a dropped Wi-Fi connection doesn't kill it. Slurm scripts come after
the MVP, as planned.

---

## 2. Package management: requirements, venv, uv

### 2.1 The concepts

- **Package**: someone else's code you import (`numpy`, `tensorflow`, `earthengine-api`).
- **Environment**: one folder holding one Python interpreter plus a set of packages.
  A **venv** ("virtual environment") is that folder. `source venv/bin/activate` makes
  `python` mean *that* Python with *those* packages. Without it, every project on the
  machine would share one set of packages and break each other.
- **`requirements.txt`**: a list of what to install. Two kinds exist, and AMW mixes them:
  - **direct, loose**: `tensorflow==2.14.0`, `torch`, `rasterio`. What *you* need. AMW's
    `requirements.txt`.
  - **lock (pinned, complete)**: every package including the packages your packages
    need, each with one exact version. AMW's `requirements-lock_2026-09-18.txt` (112
    lines). This is what makes "same code, same result" true on another machine.
- **pip** installs packages into the active environment. **uv** is a much faster
  replacement for pip + venv (written in Rust, same commands with `uv` in front). It also
  *computes* lock files for a platform other than the one you're on.

### 2.2 The problem we have today

AMW's lock file was made on a Mac and **will not install on the cluster**:
- `tensorflow-macos==2.14.0`: a Mac-only package.
- `-e git+https://github.com/facebookresearch/sam2.git@...`: an editable SAM2 checkout
  that pulls in `torch` (about 2–4 GB with GPU libraries). We don't need SAM2 or torch
  until SAM2 comes back.
- scikit-learn is used by AMW's postprocess but is missing from `requirements.txt`
  (handoff §13).

### 2.3 My recommendation (strong): uv, two input files, one lock per platform

```
requirements/detect.in        direct packages for plan/detect/assemble/persist (loose)
requirements/sam2.in          later: torch, sam2 (only on GPU nodes)
requirements/detect-linux.lock   generated: uv pip compile detect.in --python-platform linux --python-version 3.11
requirements/detect-macos.lock   generated: same, for your MacBook
```

Install on the cluster: `uv venv --python 3.11 <path>/venv` then
`uv pip sync requirements/detect-linux.lock`. `sync` installs exactly the lock and
removes anything else, so the environment can't drift.

**For:** the Mac and the cluster get the same versions of everything that exists on both;
the lock is in git, so provenance (which already records package versions) can be checked
against it; 10–100× faster installs; torch stays off the cluster until we need it.
**Against:** one new tool to learn; uv must be installed on the cluster (one file, no
admin rights needed, unless the cluster forbids it); two lock files to regenerate when a
package changes. **Alternative:** plain `python -m venv` + `pip install -r`. It works, but
pip can't make a Linux lock from your Mac, so the cluster lock would be made by hand.
**Alternative 2:** conda/mamba, if the cluster pushes it. Heavier, and it mixes its own
package versions with pip's. I'd only use it if the cluster's Python is unusable.

### 2.4 Where the environment lives

A TensorFlow venv is about 2 GB and **tens of thousands of small files**. Clusters cap
the number of files ("inodes") in `/home`, so the venv may have to go in project storage
instead. uv also keeps a download cache (`~/.cache/uv`) that should be pointed off
`/home` (`UV_CACHE_DIR`). Both depend on the quotas in the docs.

---

## 3. Getting the code and secrets onto the cluster

- **Code:** `git clone` of the repo into backed-up storage, then
  `git checkout pipeline-rebuild`. The model (16 MB) and boundaries (14 MB) are in git, so
  nothing else needs copying. A private repo needs an SSH key on the cluster added to
  GitHub (or a deploy key, read-only, for that one repo).
- **Earth Engine login without a browser.** `earthengine authenticate` opens a browser,
  which a cluster doesn't have. Two options:
  1. *User login, copied.* Run `earthengine authenticate --auth_mode=notebook` (prints a
     link, you paste back a code) once on the login node. The credentials land in
     `~/.config/earthengine/credentials` and every job on every node reads them, because
     `/home` is shared. Simple; billed as you.
  2. *Service account.* A robot Google account in the `deforestation-and-mining` project,
     registered for Earth Engine, with a JSON key file. AMW's code already supports it:
     if `GOOGLE_APPLICATION_CREDENTIALS` points at a key file, `core/gee.py:69-88` uses it.
     Doesn't depend on your personal login; the standard for unattended batch work.

  **My recommendation: start with option 1** (nothing to set up in Google Cloud) and switch
  to 2 before the long run if your login ever expires mid-job. One objection to note: you
  asked for settings in `config.yaml`, not environment variables (handoff §2.12). The key
  path is a *secret's location*, not a setting, and AMW reads it from the environment; I'd
  keep it there, or add it as `earth_engine: credentials:` in the config. Your call.
  Either way the key file must be `chmod 600` and never committed.

---

## 4. The big unknown: can compute nodes reach Earth Engine?

detect.py downloads imagery *during* scoring (one `computePixels` request per tile). That
only works if compute nodes can open HTTPS connections to `googleapis.com`. Many clusters
block outbound internet on compute nodes. Three cases:

| Case | What we do |
|---|---|
| **A. Compute nodes have internet** | Nothing changes. detect.py runs as is, inside jobs. |
| **B. Internet through a proxy** | Set `HTTPS_PROXY` in the job script. One line, no code change. |
| **C. No internet on compute nodes** | Split detect into **fetch** (on a node with internet: login, "transfer" or "data mover" node) and **score** (compute node). The imagery then *must* be stored for a while: this is the only case where we need an image cache. |

This is also the answer to your "cache images, or only store them with `utils/gee_data_pull.py`"
question:
- In case A/B: **no image cache** stays the decision (handoff §4). `gee_data_pull.py` is a
  diagnostics tool only: pulling a few tiles as GeoTIFFs to look at, or for the comparison
  thread. It writes one GeoTIFF per tile with uint16-rounded pixels (handoff §14), so
  scores from those files would differ slightly from detect.py's in-memory floats. It
  should not feed the panel.
- In case C: a *rolling* cache on scratch (fetch writes a batch, score reads and deletes
  it). It would be `.npy` files written by our own fetch step, not `gee_data_pull.py`, so
  the pixels stay float and the scores stay identical.

---

## 5. Code changes the move needs

What I can already see, smallest first. Each is small and goes through your branch as usual.

1. **`run.output_root`** in config.yaml points at the scratch run folder on the cluster.
   No code change; it's already a setting (config.py:144-150, 250).
2. **TensorFlow on Linux.** `apply_darwin_tf_compat()` (core/tf_darwin.py:19) is a Mac
   patch. Needs a check that it does nothing on Linux.
3. **Thread count.** TensorFlow and NumPy start one thread per core *of the whole node*
   (say 64), not the 4 cores Slurm gave you. 16 jobs on one node then fight over the CPU
   and everything slows down ("oversubscription"). Fix: read the allocated CPUs
   (`SLURM_CPUS_PER_TASK`, or `os.sched_getaffinity`) and pass it to
   `tf.config.threading.set_intra_op_parallelism_threads(...)` at the start of detect.py.
   About 5 lines.
4. **Wall-time and being killed.** Slurm sends `SIGTERM`, waits about 30 s, then kills the
   job. Today a kill loses only the unfinished batch (part files are written
   temp-then-rename), which is safe. Optional improvement: catch `SIGTERM`, finish
   writing the current part, record "stopped: wall time" in `_status.json`. Also: batch
   size × time per tile must fit comfortably inside the wall time.
5. **Lock file on a shared file system.** The lock the "Quarterly run and parallel layout"
   thread is building must hold across *nodes*, not just processes on one machine. A lock
   left behind by a job that was killed (node crash) must be recognisable as stale: store
   host, job ID and start time in it. I'll check that thread's version against this
   once it's pushed rather than write a second one.
6. **Provenance**: add the Slurm job ID and node name to the session entry in
   `_status.json`, so a bad part file can be traced to the job that wrote it.
7. **Floating-point differences between machines.** The Mac (ARM) and the cluster (x86,
   with Intel's oneDNN maths library on by default in TF 2.14) can give scores that
   differ in the 6th–7th decimal. Harmless for the panel, but it matters for the
   bit-identical checks and the AMW comparison. `TF_ENABLE_ONEDNN_OPTS=0` turns oneDNN
   off. Recommendation: measure it once on the pilot before deciding.

---

## 6. Storage: where, how often, and what happens on a crash

### 6.1 Where (fills in once we have the names)

| Tier | Holds | Why |
|---|---|---|
| node-local `$TMPDIR` | nothing in case A/B; the rolling image cache in case C if it's big enough | fastest, wiped at job end |
| scratch | the run folder: `plan/`, `detect/` part files, `assemble/` | big and fast, **not backed up**, often purges old files |
| project (backed up) | repo, venv, `run/config.yaml` + `provenance.json` copies, final panels, logs | survives disk failure |
| off-cluster | a copy of the final panels | survives the cluster |

**Watch out for the scratch purge.** Many clusters delete files untouched for 30–90 days.
A run that takes months could lose its early part files before assemble reads them. Two
defences: assemble each period soon after its 6 tasks finish (the period-first task order
in `quarterly_and_parallel_runs.md` §3.3 already gives this), and copy assembled period
files to project storage.

### 6.2 How often we write

Today: one part file per **500-tile batch** per (frequency, period, subregion). With
about 2.3 s per tile that's one write about every 20 minutes per process, and a crash
loses at most that. That's AMW's batch size.

- **Smaller batches** (100): lose at most 4 minutes, but 5× more files (about 385,000 for
  the full run). File-count quota is the limit here.
- **Larger batches**: fewer files, more lost per crash.

My default: keep 500, and revisit only if the measured time per tile on the cluster or the
file quota says otherwise. Assemble merges a period's parts into one file and deletes them,
so the file count peaks at roughly "parts of the periods not yet assembled".

### 6.3 What happens on each kind of crash

| What breaks | What happens today | Restart |
|---|---|---|
| one tile fails (Earth Engine 429/500, timeout) | retried up to `tries`=3 passes; then logged as failed in `_status.json` | rerun the same command: retries only the failed tiles |
| the job is killed (wall time, out of memory, node crash) | the current batch is lost; written parts are complete (atomic rename) | rerun: skips saved tiles |
| two copies run on one folder | **unsafe today**; the lock file (other thread) fixes it | – |
| scratch purged | parts gone | rerun detect for those periods |
| Earth Engine login expired | every tile fails fast | log in again, rerun |

So "restart" is always "run the same command again". With Slurm later, that's "resubmit
the array".

---

## 7. What I need from the cluster documentation

Paste whatever you have; these are the parts I'll look for. You don't need to answer
them yourself.

1. **Compute-node internet**: can jobs reach `https://earthengine-highvolume.googleapis.com`?
   Is there a proxy? (decides §4)
2. **Hardware**: CPU type (x86_64?), cores and memory per node, GPUs (for SAM2 later),
   size of node-local disk.
3. **Scheduler**: Slurm or something else; partitions/queues; maximum wall time; maximum
   jobs at once and job-array size; how interactive jobs work.
4. **Storage**: the names and paths of home / project / scratch; quota per tier in GB
   **and in number of files**; which are backed up; the scratch purge rule.
5. **Software**: module system (`module load ...`), available Python versions, whether
   uv/pip into your own folders is allowed, container support (Apptainer/Singularity).
6. **Access**: SSH login, VPN or 2FA, whether `tmux` is allowed on login nodes, how to get
   files on and off (rsync, scp, Globus, a transfer node).

## 8. Things you didn't list that we'll also need

- **Logs**: one log file per task (`<run>/logs/`), so a failure three weeks in can be read.
- **Monitoring**: `status.py` (being built in the other thread) answers "how far are we"
  without opening 840 `_status.json` files.
- **Cost of a mistake**: a first cluster run on La Pampa (80 tiles) to measure time per
  tile, memory, and whether the scores match the Mac's, before any big run.
- **Fair-share**: clusters lower your priority after heavy use; the full run's ~25,000
  core-hours (`quarterly_and_parallel_runs.md` §4) is worth checking against your
  allocation.
- **Off-cluster backup** of the final panel, and a plan for where the analysis runs
  (cluster or laptop: one period is ~18 GB in pandas, so the full panel is not a laptop job).
