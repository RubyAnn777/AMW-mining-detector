# Goethe cheat sheet: scoring runs

Copy, paste, and change only the lines marked **EDIT**. Everything after `#` on a line is a
comment, so the shell ignores it.

---

## 0. The values to change for a run

| | a) this run, 2026 | b) example: 2025 |
|---|---|---|
| `RUN` | `/work/goldmine/grambauer/runs/aca_full_2026-11-01` | same run (2025 is part of it) |
| `YEAR` | `2026` | `2025` |
| task list | `$RUN/tasks_2026_retry.txt` | `$RUN/tasks_2025.txt` |

A **run** is one folder made by `plan.py`, holding every period from 2018 to 2026. A **task** is
one period × one subregion, e.g. "monthly 2026-07, subregion 5". You don't make a new run to score
another year. You hand the same run a different task list.

---

## 1. Log in, and get ready (every new terminal)

```bash
ssh grambauer@goethe.hhlr-gu.de        # on your Mac; asks for the password
cd ~/AMW-mining-detector
source cluster/goethe_env.sh            # Python environment + Earth Engine key; prompt shows (amw-panel)
git pull                                # newest code (the job retries failed tiles by itself since PR #11)

RUN=/work/goldmine/grambauer/runs/aca_full_2026-11-01   # EDIT for another run
YEAR=2026                                                # EDIT: 2026 for a), 2025 for b)
EX=node22-006,node17-016,node17-019,node17-021,node17-024,node17-026,node19-010,node19-011,node21-022   # slow nodes to avoid; EDIT: add one when you find it slow
```
`RUN=`, `YEAR=` and `EX=` only last for this terminal. In a new terminal, set them again.

---

## 2. Where are we?

```bash
python panel_pipeline/status.py --run $RUN                         # the grid, failed tiles, progress bars
python panel_pipeline/status.py --run $RUN --frequency quarterly   # only one frequency
squeue --me                                                         # my Slurm jobs (R = running, PD = waiting)
tail -3 slurm-<JOBID>.out                                           # a job's latest lines
tail -f slurm-<JOBID>.out                                           # follow it live; Ctrl-C stops watching, not the job
sacct -j <JOBID> --format=JobID,State,Elapsed,NodeList              # a finished job: how it ended
```

---

## 3. Start scoring (or re-score failed tiles)

**After PR #13 is merged and pulled: one command does both, with no task file.** A retry is the
same command as a run. The job works out for itself what isn't done yet:
```bash
sbatch --exclude=$EX \
       cluster/detect_tasks.sbatch $RUN $YEAR 4 5       # a) YEAR=2026, b) YEAR=2025; also a period (2026-07, 2026Q3) or all
```
Until then (or if you want to see the list first), use the task-file way below.

**a) 2026: the failed tiles that are left**
```bash
python panel_pipeline/status.py --run $RUN --todo | grep -- "--from $YEAR" > $RUN/tasks_${YEAR}_retry.txt
wc -l $RUN/tasks_${YEAR}_retry.txt      # how many tasks; 0 = nothing to do
cat  $RUN/tasks_${YEAR}_retry.txt       # look at them
sbatch --exclude=$EX \
       cluster/detect_tasks.sbatch $RUN/tasks_${YEAR}_retry.txt 4 5
```

**b) 2025: a whole year** (same steps, just a different `YEAR`)
```bash
YEAR=2025
python panel_pipeline/status.py --run $RUN --todo | grep -- "--from $YEAR" > $RUN/tasks_${YEAR}.txt
wc -l $RUN/tasks_${YEAR}.txt            # a full year: 12 months × 6 + 4 quarters × 6 = 96 tasks
sbatch --exclude=$EX \
       cluster/detect_tasks.sbatch $RUN/tasks_${YEAR}.txt 4 5
```
Then submit the same `sbatch` line again for each extra node you want, **up to 6 jobs in total**
(Earth Engine's limit). Every job reads the same list, and a task that another job already has
is skipped.

What the parts mean:
- `--todo` prints a `detect.py` command for every task that's not done and not running.
  `grep -- "--from 2026"` keeps only that year's tasks. `>` writes them to a file, and `>>` would
  add them to the end of an existing file instead.
- `--exclude=$EX` keeps the job off the slow nodes listed in `EX=` (section 1). When you find a new slow one, add it there, separated by a comma with no spaces.
- `4 5` means 4 tasks at once, each with 5 download workers, so 20 Earth Engine requests per node.
- The job retries what failed by itself, up to 4 rounds with 16 minutes between rounds. Starting
  again is always safe, because saved tiles are skipped.
- **Queue behind a running job** instead of using an extra node, so it starts when that job ends:
  `sbatch --dependency=afterany:<JOBID> --exclude=$EX cluster/detect_tasks.sbatch <list> 4 5`
- **A single task, by hand on the login node** (only for a quick test, e.g. 1 tile):
  `python panel_pipeline/detect.py --run $RUN --frequency monthly --subregion 5 --from 2026-07 --to 2026-07 --limit 1`

---

## 4. Stop things

```bash
scancel <JOBID>                     # stop one job: its tasks save what they scored and unlock (takes ~1 min)
scancel --name=amw-detect           # stop ALL my scoring jobs
scancel -u $USER                    # stop EVERY job of mine, of any kind
scancel --name=amw-detect --state=PENDING   # remove only the ones still waiting in the queue
```
After a stop, a task is "partly done" or "S" in `status.py`. It continues where it stopped the
next time it's in a task list.

---

## 5. Check the scores (sanity check, ~15–30 min)

```bash
sbatch --partition=general1 --time=02:00:00 --mem=32G --cpus-per-task=2 --job-name=check \
  --output=check-%j.out \
  --wrap="source cluster/goethe_env.sh && python panel_pipeline/checks/check_detect.py --run $RUN --year $YEAR --sample 0.1"
cat check-<JOBID>.out               # when it's gone from squeue
```
(`check_detect.py` lives on the branch `claude/project-thread-s2pvh1` until it's merged:
`git fetch origin claude/project-thread-s2pvh1 && git checkout origin/claude/project-thread-s2pvh1 -- panel_pipeline/checks/check_detect.py`.)

---

## 6. My terminal sessions and processes on the login node

Goethe has several login nodes (`login01`, `login02`, ...). Your prompt shows which one you're on.
These commands only see the login node you're on now. Slurm jobs aren't here: they run on compute
nodes, so use `squeue` and `scancel` for them.

```bash
hostname                                    # which login node am I on
who | grep $USER                            # my open ssh sessions here (one line per terminal, e.g. pts/3)
tty                                         # which one of them is THIS terminal
ps -u $USER -o pid,tty,etime,cmd            # all my processes here: id, terminal, running time, command
pkill -u $USER -f check_detect.py           # stop a program by (part of) its name
kill <PID>                                  # stop one process by its id (from ps); kill -9 <PID> if it won't go
pkill -u $USER -t pts/3                     # close another ssh session of mine (its tty from `who`); not your own!
ssh login02                                 # hop to the other login node, to look there
```
In a program that's running in front of you, **Ctrl-C** stops it.

---

## 7. Leave

```bash
exit                                        # or Ctrl-D: logs out. Slurm jobs keep running.
```
Anything started with `sbatch` keeps running after you log out. Anything you started directly in
the terminal (`python ...` without sbatch) stops when the terminal closes.

---

## 8. Copy files to your Mac (run these ON THE MAC)

```bash
scp grambauer@goethe.hhlr-gu.de:/work/goldmine/grambauer/runs/aca_full_2026-11-01/plan/tiles.csv .
rsync -av --progress grambauer@goethe.hhlr-gu.de:<path on goethe> <folder on mac>/
ssh grambauer@goethe.hhlr-gu.de 'ls -l <path>/*.parquet'   # look before copying (sizes in bytes)
```

---

## 8b. Back up the run (after every scoring session)

**Why.** Goethe has three storage places with different rules:

| Place | Size | Backed up? | Use it for |
|---|---|---|---|
| `/home/goldmine/grambauer` (`~`) | 30 GB, slow | **yes, every night** | code, the key, the run's small records |
| `/work/goldmine/...` | 24 TB shared by the group, fast | **no** | the run folder while scoring |
| `/arc01`, `/arc02` | ≤ 10 TB per group, group leader requests it | no, but long-term | finished panels |

The whole run lives on `/work`, so without these steps there is exactly one copy of the results,
on the disk with no backup. The "delete after ~30 days" rule on `/work` is a request to tidy up,
not an automatic purge, but a disk failure or cleanup would lose everything.

What is in the run folder (sizes on 2026-11-09):

| Folder | Size | Can it be rebuilt? |
|---|---|---|
| `detect/` | 88 GB | only by re-scoring: days of Earth Engine time. **Most important.** |
| `panel/` | 78 GB | yes, from `detect/` with the assemble step (section 9), hours |
| `plan/`, `run/`, `logs/`, `tasks_*.txt`, `_status.json` files | < 1 GB | not exactly. Small, keep them in `/home` |
| `reports/`, `amw_comparison/` | ~20 MB | yes, by rerunning section 9 |

**a) On Goethe: small records into the backed-up home folder (seconds)**

```bash
RUN=/work/goldmine/grambauer/runs/aca_full_2026-11-01            # EDIT for another run
mkdir -p ~/runs_records/$(basename $RUN)
rsync -av --include='*/' --include='_status.json' --include='*.yaml' --include='*.json' \
  --include='tiles.csv' --include='*.txt' --include='logs/***' --exclude='*' \
  $RUN/ ~/runs_records/$(basename $RUN)/
```

**b) On the Mac: the whole run (first time: hours; afterwards only what changed)**

```bash
df -h ~                                         # need ~180 GB free for everything, ~90 GB without panel/
cd /Users/rubygrambauer/RA_Raji/AMW-mining-detector
grep -qx 'cluster_output/' .gitignore || echo 'cluster_output/' >> .gitignore   # never commit the data
rsync -av --progress --partial --exclude='_lock' \
  grambauer@goethe.hhlr-gu.de:/work/goldmine/grambauer/runs/aca_full_2026-11-01/ \
  /Users/rubygrambauer/RA_Raji/AMW-mining-detector/cluster_output/aca_full_2026-11-01/
```
- Short on space: add `--exclude='panel/'` (the panel can be rebuilt from `detect/`).
- Broke off (wifi, laptop asleep)? Run the same line again: it continues where it stopped.
- Run it again after every session: rsync only copies new and changed files, so it is quick.
- Running it while jobs are still scoring is fine; the next run picks up what was added.

**c) Once:** ask the group leader whether the group has `/arc` space
(`df -h /arc0{1,2}/archive/goldmine` on a login node). A laptop is not a backup either; the final
panels belong on `/arc` plus one copy off the cluster.

---

## 9. A finished year: panel, checks, AMW comparison, notebooks

Do this once a year is fully scored. Example: 2026. For 2025 later, change only `YEAR=`.
Every job here is safe to submit again: what is already done is skipped, other years are never touched.

**9a. Get ready** (section 1 first: log in, `source cluster/goethe_env.sh`, `git pull`, `RUN=...`)
```bash
YEAR=2026                                                     # EDIT: the year that is finished
python panel_pipeline/status.py --run $RUN | grep -A3 "^$YEAR"   # every mark of the year should be #
python panel_pipeline/status.py --run $RUN --todo | grep -c -- "--from $YEAR"   # 0 = nothing left to score
ls cluster/assemble.sbatch                                    # "No such file": run git pull first
```

**9b. AMW's files onto Goethe** (once; run the first line ON THE MAC, the second on Goethe first)
```bash
ssh grambauer@goethe.hhlr-gu.de 'mkdir -p ~/AMW-mining-detector/amw_comparison/data/amw'   # on the Mac
scp ~/RA_Raji/AMW-mining-detector/amw_comparison/data/amw/amazon_basin_Q*26_t0.4.geojson \
    grambauer@goethe.hhlr-gu.de:AMW-mining-detector/amw_comparison/data/amw/                # on the Mac
ls -l amw_comparison/data/amw/                                # on Goethe: Q126, Q226, Q326 should be there
```
AMW publishes quarters only from 2025 on, so the AMW check exists for 2025 and 2026 only.

**9c. Submit the jobs** (from `~/AMW-mining-detector`; copy all three lines at once)
```bash
A=$(sbatch --parsable cluster/assemble.sbatch $RUN $YEAR); echo "assemble job $A"
C=$(sbatch --parsable cluster/amw_compare.sbatch $RUN $YEAR); echo "compare job $C"
sbatch --dependency=afterok:$A:$C cluster/summaries.sbatch $RUN
```
What the parts mean:
- `assemble.sbatch` builds the panel for the year: months and quarters at the same time on one
  node, then runs `check_panel.py` on that year. Log: `assemble-<jobid>.out`. Estimate: 2-4 hours.
- `amw_compare.sbatch` compares our quarters with AMW's, chip by chip. Log: `compare-<jobid>.out`.
  It reads the scores directly, so it runs at the same time as the assembly.
- `summaries.sbatch` runs the four notebooks once and stores what they read from the whole run
  in `$RUN/summaries/`, so they open in seconds for you and your coauthors. Log:
  `summaries-<jobid>.out`. Add a quarter at the end (`... summaries.sbatch $RUN 2026Q2`) to
  also store that quarter's example rows and distributions.
- `--parsable` makes sbatch print only the job number; `A=$( )` keeps it in `$A`.
- `--dependency=afterok:$A:$C` = "start only after jobs `$A` and `$C` finished without error".
  Such a job shows `PD (Dependency)` in `squeue` until then. If one fails, it waits forever: `scancel` it.

**9d. Read the results**
```bash
squeue --me                                    # wait until the jobs are gone
tail -25 assemble-<JOBID>.out                  # ends with "all checks passed" and "done: assembly 0/0, checks 0"
tail -8  compare-<JOBID>.out                   # one line per subregion and quarter; "should have matched" must be 0
grep -E "done|FAILED" summaries-*.out          # one "done" line per notebook
```
Then open the notebooks in the browser (section 10), on Goethe, with nothing copied.

**9e. If something is refused**
- `not assembled: ... N tile(s) failed`: those tiles failed every try. Either score them again
  (section 3, then 9c again), or accept them as missing:
  `sbatch --export=ALL,ALLOW_FAILED=1 cluster/assemble.sbatch $RUN $YEAR` (their chips get
  `scored = False` and empty values, never 0). Decide this before assembling: a period is
  written once, and later new scores for it are refused, not mixed in.
- `detect's output changed after the ... layer was written`: that period was assembled earlier
  and its scores changed since. Nothing is overwritten by design; ask before moving anything.
- `exceeded memory limit` / `TIME LIMIT`: tell Claude the job number; the sizes in the scripts
  are estimates for the whole basin.

**Next year:** `YEAR=2025`, then 9c again. 2026 is left exactly as it is; the notebooks then show
both years.

---

## 10. Look at the results: the notebooks in the browser

**Once, after the marimo change is merged** (sets up the environment and the viewers' copy):
```bash
cd ~/AMW-mining-detector && git pull && source cluster/goethe_env.sh
uv sync --locked                                            # adds marimo to the venv, removes Jupyter; scoring packages unchanged
git clone git@github.com:RubyAnn777/AMW-mining-detector.git /work/goldmine/grambauer/amw-view
chmod -R g+rX /work/goldmine/grambauer/amw-view /work/goldmine/grambauer/venvs /work/goldmine/grambauer/python /work/goldmine/grambauer/runs
ls -ld /work/goldmine/grambauer/venvs /work/goldmine/grambauer/python /work/goldmine/grambauer/runs/*   # each should show r-x in the middle
```

The notebooks run on Goethe, next to the data. The same steps work for your coauthors (their
guide: `notebooks/README.md`).

```bash
cd /work/goldmine/grambauer/amw-view && git pull            # the viewers' copy of the code (once after each merge)
sbatch cluster/view.sbatch                                  # or: sbatch cluster/view.sbatch $RUN
cat ~/marimo-view-<JOBID>.out                               # after ~20 s: the node, the port, the address
```
In VS Code: the **Ports** tab (next to Terminal) → **Forward a Port** → type `<node>:<port>` from
the file → open the `http://localhost:...` address from the file in your browser.

To **edit** a notebook yourself (not view): in a job, `marimo edit notebooks/panel_overview.py
--host 0.0.0.0 --port <port> -- --run $RUN`, same forwarding. Changes are saved to the file;
commit them like any code.

**Updating the environment** after a change to `pyproject.toml`/`uv.lock` (after `git pull`):
```bash
uv sync --locked                       # changes the shared venv on /work in place
chmod -R g+rX /work/goldmine/grambauer/venvs /work/goldmine/grambauer/python
```
