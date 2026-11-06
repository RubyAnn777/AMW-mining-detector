#!/bin/bash
# Run detect.py tasks on ONE node, several at a time.
#
#     sbatch cluster/detect_tasks.sbatch <run folder> <year | period | all> [tasks at once] [workers per task]
#     sbatch cluster/detect_tasks.sbatch <run>/tasks.txt [tasks at once] [workers per task]
#
# With a run folder, the job makes its own list when it starts: every task of that year
# (2026), period (2026-07, 2026Q3) or of the whole run (all) that is not done and not
# running (status.py --todo). So the same command starts a year and retries it: what is
# done is left out, failed tiles and stopped tasks are picked up. Saved in
# <run>/logs/tasks_<jobid>.txt. A task file instead (status.py --todo > file) runs
# exactly those lines.
#
# Each line of the file is one detect.py command (one period, one subregion). The job
# runs N of them side by side (default 4), each with --workers W (default 5), and
# starts the next line whenever one finishes, until the file is done. Earth Engine
# sees N x W requests at once: keep that near the project's limit of 40 (README B4).
#
# Measured 28 Oct - 1 Nov 2026 (README B4): the node's scoring is the limit, not Earth
# Engine. 4 tasks x 10 TensorFlow threads score about 115-145 tiles/min per node, 2 x 20
# only 88. 3 workers per task barely kept the scorer fed (downloads take ~5 s); 5 do.
# Leftovers are retried by the job itself, in rounds. When the list is done, the job asks
# `status.py --todo` what is still missing, keeps only tasks that were in its own list
# (so a 2026 list stays 2026), and runs those: tasks with failed tiles score only those
# tiles, and tasks that died without cleaning up ("S") continue where they stopped.
# Before each extra round it waits 16 min, so a dead task's lock is stale (15 min) and
# Earth Engine's busy spell has passed. It stops when nothing is left, after ROUNDS
# rounds in all (default 4; `sbatch --export=ALL,ROUNDS=6 ...` to change), or on a stop.
# A finished task is skipped in seconds, so resubmitting is always safe too. Stop with
# `scancel <jobid>`: every task gets the stop signal, saves what it scored, and gives
# its lock back.
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
#SBATCH --time=7-00:00:00            # Goethe cancels jobs over 7 days when it patches nodes (wiki); tasks resume after it
#SBATCH --signal=B:USR1@300          # SIGUSR1 to this script 5 min before the limit
#SBATCH --output=slurm-%j.out        # the job's own log, next to where sbatch was run

USAGE="usage: sbatch cluster/detect_tasks.sbatch <run folder> <year|period|all> [tasks at once] [workers]
   or: sbatch cluster/detect_tasks.sbatch <tasks.txt> [tasks at once] [workers]"
FROM_RUN=""
if [ -d "${1:?${USAGE}}" ]; then                  # a run folder: make the list below
    FROM_RUN="$1"
    WHICH="${2:?${USAGE}}"
    shift
fi
TASKS="$1"
AT_ONCE="${2:-4}"
WORKERS="${3:-5}"
ROUNDS="${ROUNDS:-4}"                         # the list once, then up to 3 rounds of leftovers
ROUND_WAIT_S="${ROUND_WAIT_S:-960}"                             # 16 min: longer than a lock takes to go stale
STOPPING=0
DETECT='^python panel_pipeline/detect.py --run'   # how a running task's command line starts

export TF_THREADS=$(( 40 / AT_ONCE ))         # share the node's 40 cores between the tasks
source cluster/goethe_env.sh               # run sbatch from the repository root

stop_tasks() {
    # No new task (xargs ends), and every running task saves, unlocks and exits.
    # detect.py treats a repeated SIGTERM as the same request, so forwarding it is safe.
    echo "$(date -u +%FT%TZ) $1: no new task starts; every running task saves and stops"
    STOPPING=1                                   # and no further round
    kill -TERM "${XARGS_PID}" "${SLEEP_PID}" 2>/dev/null
    pkill -TERM -u "${USER}" -f "${DETECT}"
}
trap 'stop_tasks "time limit in 5 min"' USR1    # from Slurm, --signal above
trap 'stop_tasks "cancelled"' TERM              # from scancel

if [ -n "${FROM_RUN}" ]; then
    mkdir -p "${FROM_RUN}/logs"
    TASKS="${FROM_RUN}/logs/tasks_${SLURM_JOB_ID}.txt"
    FILTER="--from ${WHICH} "                   # a period: status.py prints "--from <tag> --to <tag>"
    [[ "${WHICH}" =~ ^[0-9]{4}$ ]] && FILTER="--from ${WHICH}"   # a year: its months and quarters
    [ "${WHICH}" = all ] && FILTER="--from "
    python panel_pipeline/status.py --run "${FROM_RUN}" --todo | grep -F -- "${FILTER}" > "${TASKS}" || true
    if [ "$(grep -c . "${TASKS}")" = 0 ]; then
        echo "$(date -u +%FT%TZ) nothing to do in ${FROM_RUN} for ${WHICH}: every task is done or running"
        exit 0
    fi
fi

echo "$(date -u +%FT%TZ) job ${SLURM_JOB_ID} on $(hostname): ${AT_ONCE} tasks at once, ${WORKERS} workers each, ${TF_THREADS} threads each"
echo "task list: ${TASKS} ($(grep -c . "${TASKS}") lines)"

run_list() {
    # xargs starts AT_ONCE shells, hands each the next line of the file ($1) as the
    # command to run, and keeps going until the file is empty. Each task's own log is
    # its folder's _log.txt; its screen output goes to this job's slurm-<id>.out,
    # prefixed with its period (sed -u: line by line; without it sed holds about 4 KB
    # back when writing to a file).
    grep . "$1" | xargs -P "${AT_ONCE}" -I '{}' bash -c \
        'line="{}"; tag=$(echo "$line" | sed -E "s/.*--from ([^ ]+).*--subregion ([0-9]+).*/\1 s\2/; s/.*--subregion ([0-9]+).*--from ([^ ]+).*/\2 s\1/"); \
         echo "$(date -u +%FT%TZ) start  $tag"; \
         $line --workers '"${WORKERS}"' 2>&1 | sed -u "s/^/[$tag] /"; \
         echo "$(date -u +%FT%TZ) end    $tag (exit ${PIPESTATUS[0]})"' &
    XARGS_PID=$!
    wait "${XARGS_PID}"          # returns early when a trapped signal arrives ...
    wait "${XARGS_PID}"          # ... so wait once more for xargs itself
}

leftovers() {
    # What is still missing ($1: where to write it): status.py's --todo, only the lines
    # that are also in this job's own task list. Prints how many there are.
    python panel_pipeline/status.py --run "${RUN_DIR}" --todo \
        | grep -Fx -f "${TASKS}" > "$1" || true
    grep -c . "$1" || true
}

# The run folder, as the task list's commands name it (status.py prints it the same way).
RUN_DIR=$(grep -m1 -oE -- '--run [^ ]+' "${TASKS}" | cut -d' ' -f2)
mkdir -p "${RUN_DIR}/logs"
LIST="${TASKS}"
for ROUND in $(seq 1 "${ROUNDS}"); do
    [ "${ROUND}" -gt 1 ] && echo "$(date -u +%FT%TZ) round ${ROUND} of ${ROUNDS}: $(grep -c . "${LIST}") task(s) left over"
    run_list "${LIST}"
    [ "${STOPPING}" = 1 ] || [ "${ROUND}" = "${ROUNDS}" ] && break
    NEXT="${RUN_DIR}/logs/leftovers_${SLURM_JOB_ID}_round$(( ROUND + 1 )).txt"
    if [ "$(leftovers "${NEXT}")" = 0 ]; then
        echo "$(date -u +%FT%TZ) nothing of this list is left to do"
        break
    fi
    echo "$(date -u +%FT%TZ) $(grep -c . "${NEXT}") task(s) not done yet; next round in $(( ROUND_WAIT_S / 60 )) min"
    sleep "${ROUND_WAIT_S}" & SLEEP_PID=$!
    wait "${SLEEP_PID}"          # a stop signal ends the wait at once
    [ "${STOPPING}" = 1 ] && break
    if [ "$(leftovers "${NEXT}")" = 0 ]; then   # again: other jobs may have done them
        echo "$(date -u +%FT%TZ) nothing of this list is left to do"
        break
    fi
    LIST="${NEXT}"
done

for _ in $(seq 60); do       # and up to 5 min for tasks still saving (they are xargs'
    pgrep -u "${USER}" -f "${DETECT}" > /dev/null || break   # grandchildren, not ours to wait on)
    sleep 5
done

echo "$(date -u +%FT%TZ) job ${SLURM_JOB_ID} finished"
