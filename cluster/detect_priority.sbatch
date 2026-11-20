#!/bin/bash
# Two-stage scoring: run detect.py tasks for one priority's tiles only, on ONE node.
#
#     sbatch cluster/detect_priority.sbatch <run folder> 1 <year | period | all> [tasks at once] [workers per task]
#     sbatch cluster/detect_priority.sbatch <run folder> 2 <year | period | all> [tasks at once] [workers per task]
#
# Priority 1 = the tiles in which AMW has ever detected mining, plus rings of neighbours;
# priority 2 = all the others (the run's list: panel_pipeline/priority.py, made once).
# Both write into the same task folders as cluster/detect_tasks.sbatch, so priority 2
# skips every tile priority 1 saved, and a task that has had both passes is exactly a
# task scored in one pass. The normal job (detect_tasks.sbatch) is unchanged and can
# still be used at any time: it scores whatever is missing, of either priority.
#
# Everything else (task list from status.py, N tasks side by side, retry rounds, stop
# and time-limit handling) is detect_tasks.sbatch's; this script only sets PRIORITY and
# hands over to it, in the same process.
#
#SBATCH --job-name=amw-prio
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%j.out

USAGE="usage: sbatch cluster/detect_priority.sbatch <run folder> <1|2> <year|period|all> [tasks at once] [workers]"
RUN="${1:?${USAGE}}"
PRIORITY="${2:?${USAGE}}"
WHICH="${3:?${USAGE}}"
case "${PRIORITY}" in 1|2) ;; *) echo "${USAGE}"; exit 2 ;; esac
[ -f "${RUN%/}/priority/record.json" ] || {
    echo "no priority list in ${RUN}: python panel_pipeline/priority.py --run ${RUN} --rings 3"; exit 2; }
export PRIORITY
# exec: the same process, so Slurm's signals reach detect_tasks.sbatch's traps
exec bash cluster/detect_tasks.sbatch "${RUN}" "${WHICH}" "${@:4}"
