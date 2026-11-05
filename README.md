<!-- PROVENANCE: FORK-MODIFIED — upstream file modified in this fork; upstream original: earthrise-media/mining-detector -->
# AMW Mining Detector: research fork

This is a fork of [earthrise-media/mining-detector](https://github.com/earthrise-media/mining-detector)
(Amazon Mining Watch, "AMW"). We use AMW's model to score every small square of the
Amazon ("chip") for gold mining, **month by month and quarter by quarter from 2018**, and
save the scores as a **panel**: one row per chip and period, with quality columns next to
each score. Nothing is published to a website.

This page is the short, step-by-step guide to running it. Every option and every
output column is explained in the full reference,
[`panel_pipeline/README.md`](panel_pipeline/README.md).

Contents

1. [The flow in one picture](#1-the-flow-in-one-picture)
2. [How to read a command](#3-how-to-read-a-command)
3. [A run on the Goethe cluster, step by step](#4-a-run-on-the-goethe-cluster-step-by-step)
4. [The same on a laptop](#5-the-same-on-a-laptop)
5. [All commands on one page](#6-all-commands-on-one-page)
6. [When something goes wrong](#7-when-something-goes-wrong)
7. [What is where in this repository](#8-what-is-where-in-this-repository)

---

## 1. The flow in one picture

```
 a config file        plan.py            status.py --todo      detect.py (many times)      assemble_period.py
 (the settings)  ──►  (once per run)  ──►  (the task list)  ──►  (the scoring, in a     ──►  (one tidy file
                                                                 Slurm job)                  per period)
                                              ▲                        │
                                              └── status.py: how far? ─┘
```

In words:

1. **Config.** You write down every setting of the run in one YAML file.
2. **Plan.** `plan.py` reads it once, makes the run's folder, freezes a copy of the
   settings there, and lists every tile and chip that will be scored. No Earth Engine yet.
3. **Task list.** `status.py --todo` writes one `detect.py` command per (period,
   subregion) that still has to be done.
4. **Detect.** Each `detect.py` command downloads the satellite pictures for its tiles
   from Earth Engine, runs AMW's model on them, and saves the scores every few minutes.
   On the cluster, a job script works through the task list, several tasks at a time.
5. **Status.** At any time, `status.py` shows what is done, running, or broken.
6. **Assemble.** When every subregion of a period is done, `assemble_period.py` joins
   them into one file per period, with AMW's detection flags added. This is the panel.

---

## 2. How to read a command

```bash
python panel_pipeline/detect.py --run /work/goldmine/me/runs/aca --frequency quarterly --subregion 2
#      └─ the script to run ──┘  └──── options: --name value, --name value, ... ────────────────┘
```

- `python <file>.py` runs that Python file. Always run commands **from the repository
  folder** (`~/AMW-mining-detector`), because the paths start there.
- Everything after it are **options** (also called flags or arguments). Each is
  `--name value`. Some take no value and work like a switch (`--todo`).
  The order of the options does not matter.
- Every script explains its own options: add `--help`, for example
  `python panel_pipeline/detect.py --help`.
- A `\` at the end of a line means "the command continues on the next line".
- `#` starts a comment: the shell ignores the rest of the line.
- `> file.txt` saves what a command prints into a file instead of the screen.
- `$RUN` is a **shell variable**: after `RUN=/some/path`, typing `$RUN` is the same as
  typing `/some/path`. It is forgotten when the terminal closes.

---

Run this to set up the environment when not done!

# Mac
uv pip sync requirements/panel-macos.txt
# Goethe
uv pip sync requirements/panel-linux.txt


## 3. A run on the Goethe cluster, step by step

One-time setup (account, SSH key, Python environment, Earth Engine key) is in
[`panel_pipeline/README.md` B2](panel_pipeline/README.md#b2-one-time-setup). The steps
below assume it is done.

### Step 0. Every time you log in

```bash
ssh <user>@goethe.hhlr-gu.de              # from your laptop's terminal or use SSH login in VS Code
cd ~/AMW-mining-detector                  # the repository can also be chosen in VS Code's "Open Folder"
source cluster/goethe_env.sh              # turns on the Python environment, the Earth Engine key, thread limits
RUN=/work/goldmine/<user>/runs/<run>      # $RUN now means your run's folder - set it before 
squeue --me                               # which of your jobs are running (an empty list is an answer too)
```

### Step 1. Make a config

```bash
cp panel_pipeline/config.yaml configs/<run>.yaml      # copy the template
nano configs/<run>.yaml                               # edit it (any editor), or just edit in VS Code
python panel_pipeline/lib/config.py --config configs/<run>.yaml   # check it
```

What to change in the copy:

| Setting | What to put |
|---|---|
| `run.name` | The run's name, also its folder name. Letters, digits, `-`, `_`. Convention: `<area>_<YYYY-MM-DD>`, for example `aca_full_2026-10-23`. |
| `run.output_root` | Where run folders go. On Goethe: `/work/goldmine/<user>/runs`. On a laptop: `tests_output`. |
| `periods.first_month`, `periods.last_month` | The first and last month to score, like `2018-01` and `2026-09`. For quarters, these must cover whole quarters. |
| `periods.frequencies` | `[monthly, quarterly]`, or just one of them. |
| `region.clip_to` | `null` for the whole Amazon, or a GeoJSON file (like `data/boundaries/test_region.geojson`) to score only a small test area. |
| `imagery.max_workers` | How many downloads one task runs at once. On the cluster you set this per job instead (step 5). |

Leave the rest as it is (model, thresholds, AMW's imagery settings) unless you mean to
change the method. The check command prints a summary and `config OK`, or a
`CONFIG ERROR` that names the problem (typo, missing setting, wrong value).

**Commit the config** (`git add configs/<run>.yaml && git commit -m "config for <run>"`)
before planning, so the run records exactly which code and settings made it. Example
configs: `configs/aca_full_2026-10-23.yaml` (whole Amazon), `configs/test_region_2026-10-19.yaml`
(small test).

### Step 2. Plan the run (once)

A small test area plans in seconds on the login node:

```bash
python panel_pipeline/plan.py --config configs/<run>.yaml
```

The whole Amazon takes 2 to 3 hours, so do it inside a job instead:

```bash
salloc --partition=general1 --nodes=1 --time=4:00:00 --mem=0   # ask for a compute node and wait for it
srun python panel_pipeline/plan.py --config configs/<run>.yaml # run plan.py on that node
exit                                                           # give the node back
```

`salloc` asks Slurm for a node and waits until you get it; `srun` runs the command on
that node; `exit` ends the allocation. `plan.py` prints how many tiles and chips the run
has, and `planned -> <folder>` at the end.

Good to know:

- A run is planned **once**. Planning the same `run.name` again stops with
  `PLAN ERROR: ... already exists`. A different setting means a new run with a new name.
- `plan.py` saves a copy of the config inside the run (`$RUN/run/config.yaml`). Every
  later command reads **that copy**, so editing `configs/<run>.yaml` afterwards changes
  nothing.
- A `WARNING ... differ from the last commit` means you forgot to commit (step 1).

### Step 3. Make the task list

```bash
python panel_pipeline/status.py --run $RUN --todo > $RUN/tasks.txt
head -3 $RUN/tasks.txt            # look at the first three lines
wc -l $RUN/tasks.txt              # how many tasks
```

Each line is one complete `detect.py` command, for example:

```
python panel_pipeline/detect.py --run /work/goldmine/me/runs/aca --frequency monthly --subregion 1 --from 2018-01 --to 2018-01
```

The list is in **period order** (2018-01 subregions 1 to 6, then 2018-02, ...), and it
leaves out tasks that are already done or running. Add `--frequency quarterly` to list
only quarterly tasks (or `monthly`). To score only some periods first, cut the file
down, for example `grep "2024Q" $RUN/tasks.txt > $RUN/tasks_2024.txt`.

### Step 4. Test one task by hand

Before leaving the cluster alone for days, watch one task run for a few minutes:

```bash
salloc --partition=test --nodes=1 --time=1:55:00 --mem=0
srun python panel_pipeline/detect.py --run $RUN --frequency quarterly --subregion 6 \
     --from 2024Q3 --to 2024Q3 --workers 20 --limit 200
exit
```

The `detect.py` options:

| Option | Meaning |
|---|---|
| `--run` | the run folder (required) |
| `--frequency` | `monthly` or `quarterly` (required) |
| `--subregion` | 1 to 6 (required) |
| `--from`, `--to` | first and last period, like `2024Q3`. `--from X --to X` is exactly one task. Leave both out to do every period of the run. |
| `--workers` | downloads at once in this task. The main speed setting (see step 5). |
| `--limit` | score only the first N tiles: a test. The task then shows as "partly done" and the real run finishes it. |

The rest (`--batch-tiles`, `--flush-minutes`, `--tries`, `--mac-eager`, `--image-cache`)
are explained in [`panel_pipeline/README.md` A4](panel_pipeline/README.md#a4-detectpy-the-scoring).

Every few minutes it prints a progress line, like:

```
2024Q3  try 1: 44/200 tiles saved (+22, 0 failed) | 11 tiles/min, download median 101 s
(slowest 106 s), workers busy 82%, scoring busy 18%, 0 retries | 20 downloading, ...
```

Check that the "tiles saved" number grows, and that it says `0 failed` and `0 retries`. Ctrl-C stops it safely (what
is scored is kept).

### Step 5. Start the real run

```bash
sbatch cluster/detect_tasks.sbatch $RUN/tasks.txt 4 5
#                                  └─ task list ┘ │ └─ downloads per task (W)
#                                                 └─ tasks at once on the node (N)
```

Run this **from the repository folder**. `sbatch` hands the job to Slurm and returns
at once with `Submitted batch job <id>`. The job then runs on one compute node for up
to 7 days. It works through `tasks.txt` from top to bottom, N tasks side
by side, and starts the next line whenever one finishes. `4 5` is the measured best
shape for a 40-core node: four tasks of 10 TensorFlow threads, five downloads each,
about 115–145 tiles a minute; three such nodes use Earth Engine's 40 requests at once
([`panel_pipeline/README.md` B4](panel_pipeline/README.md#b4-speed-and-limits-read-before-planning-a-big-run)).

### Step 6. Watch it

```bash
squeue --me                                              # is the job still running? (ST = R means running)
python panel_pipeline/status.py --run $RUN               # how far is the run?
tail -f slurm-<id>.out                                   # the job's own log, live (Ctrl-C stops watching, not the job)
tail -3 $RUN/detect/quarterly/period=2024Q3/subregion=6/_log.txt   # one task's log
```

`status.py` prints one line per year and one mark per subregion:

```
#  done   R  running   !  has failed tiles   ~  partly done   S  stopped without cleaning up   .  not started
```

followed by the tasks worth looking at, with their first error. It only reads, so it is
safe to run as often as you like, on the login node.

### Step 7. Stop, resume, restart

- **Stop:** `scancel <id>`. Every running task saves what it has scored and stops.
- **Resume or restart** (after `scancel`, a crash, or the 7-day limit, before which the
  job stops its tasks cleanly by itself): make a fresh task list and submit again.
  ```bash
  python panel_pipeline/status.py --run $RUN --todo > $RUN/tasks.txt
  sbatch cluster/detect_tasks.sbatch $RUN/tasks.txt 4 5
  ```
  Finished tasks are not in the new list; half-finished ones continue where they stopped
  (saved tiles are skipped, failed tiles retried). Nothing is scored twice, so
  resubmitting is always safe.
- **Leftovers are retried by the job itself.** When a job has worked through its list,
  it asks `status.py --todo` what is still missing from that list (failed tiles, `S`
  tasks that died without cleaning up), waits 16 min, and runs those, up to 4 rounds in
  all. Only a task that still fails after that, or one left behind by a job that ended
  completely, needs a new task list and `sbatch`. `status.py` shows its error.

### Step 8. Assemble the panel

When every subregion of a period is `#` (done), join them into one file per period. A
full-Amazon period needs 10 to 20 GB of memory, so do it in a job:

```bash
salloc --partition=general1 --nodes=1 --time=4:00:00 --mem=0
srun python panel_pipeline/assemble_period.py --run $RUN --frequency quarterly
exit
```

| Option | Meaning |
|---|---|
| `--run`, `--frequency` | which run, which frequency (both required) |
| `--from`, `--to` | only these periods (default: all of the run's) |
| `--allow-failed` | also assemble periods with tiles that failed every try; their chips get empty scores and `scored = False` |

It assembles every period that is ready, skips periods already assembled, and at the
end lists the periods it could not do and why. So you can run it again whenever more
periods are done. The output, per period:

```
$RUN/panel/quarterly/scores/period=2024Q3/data.parquet        the scores and quality columns of every chip
$RUN/panel/quarterly/postprocess/period=2024Q3/data.parquet   AMW's detection flags and distances, same rows
```

### Step 9. Look at the results, or copy them to a laptop

Read them in Python (inside a job on Goethe, or on a laptop):

```python
import pandas as pd
run = "/work/goldmine/<user>/runs/<run>"     # or the laptop copy, e.g. "tests_output/<run>"
d = pd.read_parquet(f"{run}/panel/quarterly/scores", columns=["chip_id", "period", "confidence", "n_clear"])
```

Copy to a laptop with `rsync`, **typed on the laptop**. It copies only what is new, so
it can be run again at any time:

```bash
rsync -av <user>@goethe.hhlr-gu.de:/work/goldmine/<user>/runs/<run>/{panel,plan,run} tests_output/<run>/
```

What every column means, more reading examples, and how big things are:
[`panel_pipeline/README.md` A6 and B6](panel_pipeline/README.md#a6-reading-the-output).

---

## 5. The same on a laptop

The commands are the same; there is just no Slurm. Run each one directly in the
terminal, from the repository folder, with the environment turned on
(`source venv/bin/activate`; setting it up: [`panel_pipeline/README.md` A1](panel_pipeline/README.md#a1-environment)).

```bash
python panel_pipeline/lib/config.py --config configs/test_region_2026-10-19.yaml
python panel_pipeline/plan.py      --config configs/test_region_2026-10-19.yaml
python panel_pipeline/detect.py    --run tests_output/test_region_2026-10-19 --frequency quarterly --subregion 2 --from 2026Q1 --to 2026Q3
python panel_pipeline/status.py    --run tests_output/test_region_2026-10-19
python panel_pipeline/assemble_period.py --run tests_output/test_region_2026-10-19 --frequency quarterly
```

Differences from Goethe: `run.output_root` is `tests_output` (ignored by git); Earth
Engine login is your own (`earthengine authenticate`, once); keep areas small, since a
laptop stops when it sleeps. Ctrl-C stops a task safely, and running the same command
again resumes it.

---

## 6. All commands on one page

| What | Command | Where |
|---|---|---|
| Start a Goethe session | `ssh <user>@goethe.hhlr-gu.de`, then `cd ~/AMW-mining-detector`, `source cluster/goethe_env.sh`, `RUN=...` | laptop, then login node |
| Check the Earth Engine login | `python cluster/check_login.py` | login node |
| Make a config | `cp panel_pipeline/config.yaml configs/<run>.yaml` and edit | login node |
| Check a config | `python panel_pipeline/lib/config.py --config configs/<run>.yaml` | login node |
| Plan a run | `python panel_pipeline/plan.py --config configs/<run>.yaml` (in `salloc`/`srun` for a big area) | job |
| Make the task list | `python panel_pipeline/status.py --run $RUN --todo > $RUN/tasks.txt` | login node |
| Test one task | `srun python panel_pipeline/detect.py --run $RUN --frequency ... --subregion ... --from ... --to ... --limit 200` | `salloc` on `test` |
| Start the run | `sbatch cluster/detect_tasks.sbatch $RUN/tasks.txt 2 20` | login node |
| My jobs | `squeue --me` | login node |
| Stop a job | `scancel <id>` | login node |
| How far is the run | `python panel_pipeline/status.py --run $RUN` | login node |
| A job's log | `tail -f slurm-<id>.out` | login node |
| Assemble | `srun python panel_pipeline/assemble_period.py --run $RUN --frequency quarterly` | job |
| Copy results | `rsync -av <user>@goethe.hhlr-gu.de:<path>/ <local path>/` | laptop |
| Help for any script | `python <script>.py --help` | anywhere |

---

## 7. When something goes wrong

| You see | Why | What to do |
|---|---|---|
| `python: command not found`, or `ModuleNotFoundError` | the environment is not on | `source cluster/goethe_env.sh` (laptop: `source venv/bin/activate`) |
| `CONFIG ERROR: ...` | a typo or wrong value in the config | fix what the message names, check again |
| `PLAN ERROR: ... already exists` | this run was planned before | choose a new `run.name` |
| `FileNotFoundError: GOOGLE_APPLICATION_CREDENTIALS=...` | the Earth Engine key is not where `goethe_env.sh` says | `ls -l ~/.secrets/`; fix the path in `cluster/goethe_env.sh` |
| many `retries` in the progress line, tiles failing with `429` | too many downloads at once | smaller N × W (step 5) |
| `S` in `status.py` | a task died without cleaning up | nothing special: it is in the next `--todo` list and continues |
| `SKIPPED: ... is being written by ...` | two copies of the same task | let the first one finish; no harm done |
| the job ended at its time limit | 7 days (Goethe's patch rule); the job stopped its tasks cleanly 5 min before | nothing is lost: step 7 |
| one job's `download median` is 20–30 s and its `scoring busy` 20%, the others' 5 s and 90% | that node or its network link | `scancel --batch --signal=USR1 <id>` (clean stop), then step 7 with `sbatch --exclude=<node> ...` |

More cases: [`panel_pipeline/README.md` B7](panel_pipeline/README.md#b7-if-something-breaks).

---

## 8. What is where in this repository

```
panel_pipeline/    OUR PIPELINE. The steps you run:
  config.yaml          the settings template (copy it to configs/)
  plan.py              step 2
  detect.py            step 4/5, the scoring
  status.py            progress and the task list
  assemble_period.py   step 8
  lib/                 helper code the steps use (only lib/config.py is run on its own)
  checks/              speed tests and a notebook for looking at runs; never writes a run
  docs/                everything written down: decisions, findings, explanations (start at docs/README.md)
  README.md            the full reference for every option and column
cluster/           Goethe scripts: goethe_env.sh, detect_tasks.sbatch, check_login.py
configs/           one committed config per run
requirements/      the exact Python packages, per platform
amw_comparison/    our scores compared with AMW's published ones (own README)
models/            AMW's model (the CNN ensemble in use); tracked in git, no download needed
data/boundaries/   the region files (Amazon_ACA subregions, test areas, Andes)
core/ scripts/ utils/   AMW's own code. We import four files from core/; AMW's own
                   driver (scripts/pipeline.py) and its website publishing are not used
```

**Provenance.** Files we wrote or changed start with a `PROVENANCE: FORK-ORIGINAL` or
`FORK-MODIFIED` line; untagged files are AMW's, unchanged. The full list is
`docs/PROVENANCE.md` (`python scripts/provenance_report.py` regenerates it). AMW's own
README: [github.com/earthrise-media/mining-detector](https://github.com/earthrise-media/mining-detector#readme).
