#!/bin/bash
# Run detect.py tasks from a task list on ONE node, several at a time.
#
#     python panel_pipeline/status.py --run <run> --todo > <run>/tasks.txt
#     sbatch cluster/detect_tasks.sbatch <run>/tasks.txt [tasks at once] [workers per task]
#
# Each line of the file is one detect.py command (one period, one subregion). The job
# runs N of them side by side (default 2), each with --workers W (default 20), and
# starts the next line whenever one finishes, until the file is done. Earth Engine
# sees N x W requests at once: keep that at or under the project's limit (README B4).
# A task that fails stays in the next `status.py --todo`; a finished one is skipped
# in seconds, so resubmitting is always safe. Stop with `scancel <jobid>`: every task
# gets the stop signal, saves what it scored, and gives its lock back.
#
# Near the time limit, Slurm signals only this script (--signal=B:..., five minutes
# early), not the tasks; on 25 Oct 2026 a task killed at its limit never got the
# signal and left a stale lock. So the script forwards it: no new task starts, every
# running task saves and stops, and the script waits for them before it ends.
#
#SBATCH --job-name=amw-detect
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0                      # all of the node's memory (the node is ours anyway)
#SBATCH --time=21-00:00:00           # the partition's maximum; tasks resume after it
#SBATCH --signal=B:USR1@300          # SIGUSR1 to this script 5 min before the limit
#SBATCH --output=slurm-%j.out        # the job's own log, next to where sbatch was run

TASKS="${1:?usage: sbatch cluster/detect_tasks.sbatch <tasks.txt> [tasks at once] [workers]}"
AT_ONCE="${2:-2}"
WORKERS="${3:-20}"
DETECT='^python panel_pipeline/detect.py --run'   # how a running task's command line starts

export TF_THREADS=$(( 40 / AT_ONCE ))         # share the node's 40 cores between the tasks
source cluster/goethe_env.sh               # run sbatch from the repository root

stop_tasks() {
    # No new task (xargs ends), and every running task saves, unlocks and exits.
    # detect.py treats a repeated SIGTERM as the same request, so forwarding it is safe.
    echo "$(date -u +%FT%TZ) $1: no new task starts; every running task saves and stops"
    kill -TERM "${XARGS_PID}" 2>/dev/null
    pkill -TERM -u "${USER}" -f "${DETECT}"
}
trap 'stop_tasks "time limit in 5 min"' USR1    # from Slurm, --signal above
trap 'stop_tasks "cancelled"' TERM              # from scancel

echo "$(date -u +%FT%TZ) job ${SLURM_JOB_ID} on $(hostname): ${AT_ONCE} tasks at once, ${WORKERS} workers each, ${TF_THREADS} threads each"
echo "task list: ${TASKS} ($(grep -c . "${TASKS}") lines)"

# xargs starts AT_ONCE shells, hands each the next line as the command to run, and
# keeps going until the file is empty. Each task's own log is its folder's _log.txt;
# its screen output goes to this job's slurm-<id>.out, prefixed with its period
# (sed -u: line by line; without it sed holds about 4 KB back when writing to a file).
grep . "${TASKS}" | xargs -P "${AT_ONCE}" -I '{}' bash -c \
    'line="{}"; tag=$(echo "$line" | sed -E "s/.*--from ([^ ]+).*--subregion ([0-9]+).*/\1 s\2/; s/.*--subregion ([0-9]+).*--from ([^ ]+).*/\2 s\1/"); \
     echo "$(date -u +%FT%TZ) start  $tag"; \
     $line --workers '"${WORKERS}"' 2>&1 | sed -u "s/^/[$tag] /"; \
     echo "$(date -u +%FT%TZ) end    $tag (exit ${PIPESTATUS[0]})"' &
XARGS_PID=$!

wait "${XARGS_PID}"          # returns early when a trapped signal arrives ...
wait "${XARGS_PID}"          # ... so wait once more for xargs itself
for _ in $(seq 60); do       # and up to 5 min for tasks still saving (they are xargs'
    pgrep -u "${USER}" -f "${DETECT}" > /dev/null || break   # grandchildren, not ours to wait on)
    sleep 5
done

echo "$(date -u +%FT%TZ) job ${SLURM_JOB_ID} finished the list"
