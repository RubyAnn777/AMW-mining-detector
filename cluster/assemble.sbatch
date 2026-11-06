#!/bin/bash
# Assemble one year of the panel on ONE node, then check it.
#
#     sbatch cluster/assemble.sbatch <run folder> <year>
#     sbatch --export=ALL,ALLOW_FAILED=1 cluster/assemble.sbatch <run folder> <year>
#
# Runs assemble_period.py for the year's months and the year's quarters side by side (two
# processes on the node), then check_panel.py on that year. Every period is written once
# and never changed, so:
#   - running it again for the same year only fills in what is missing ("already
#     assembled" for the rest), and
#   - another year is the same command with another <year>; the years already in the
#     panel are not touched. Several years can run at once, one job each: every period
#     has its own lock, so two jobs never write the same period.
# A period is refused (and named at the end of the log) while one of its subregions is
# not finished or has failed tiles. ALLOW_FAILED=1 assembles it anyway: the failed
# tiles' chips get a row with scored = False and empty values (README A7).
#
# Memory: one period of all of Amazon_ACA is about 25-30 GB (estimate), two at once
# ~60 GB; a general1 node has 192 GB. Time: about 10-20 min per period (estimate:
# measured 40 s for 10.5 M chips, the region has ~145 M), so a year of 12 months and
# 4 quarters takes ~2-4 h with months and quarters side by side.
#
#SBATCH --job-name=amw-assemble
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0                      # all of the node's memory (the node is ours anyway)
#SBATCH --time=1-00:00:00
#SBATCH --output=assemble-%j.out     # the job's log, next to where sbatch was run

set -o pipefail
USAGE="usage: sbatch cluster/assemble.sbatch <run folder> <year>"
RUN="${1:?${USAGE}}"
YEAR="${2:?${USAGE}}"
[ -f "${RUN}/run/config.yaml" ] || { echo "ASSEMBLE ERROR: ${RUN} is not a planned run"; exit 2; }
[[ "${YEAR}" =~ ^[0-9]{4}$ ]] || { echo "ASSEMBLE ERROR: <year> must be like 2026, not '${YEAR}'"; exit 2; }

source cluster/goethe_env.sh               # run sbatch from the repository root
EXTRA=""
[ "${ALLOW_FAILED:-0}" = "1" ] && EXTRA="--allow-failed"

echo "$(date '+%F %T')  assembling ${YEAR} of ${RUN} on $(hostname), code $(git rev-parse --short HEAD) ${EXTRA}"
# Months and quarters in parallel; each line of the log starts with which one it is.
python -u panel_pipeline/assemble_period.py --run "${RUN}" --frequency monthly \
       --from "${YEAR}-01" --to "${YEAR}-12" ${EXTRA} 2>&1 | sed -u 's/^/[monthly]   /' &
MONTHLY=$!
python -u panel_pipeline/assemble_period.py --run "${RUN}" --frequency quarterly \
       --from "${YEAR}Q1" --to "${YEAR}Q4" ${EXTRA} 2>&1 | sed -u 's/^/[quarterly] /' &
QUARTERLY=$!
wait ${MONTHLY};   M=$?
wait ${QUARTERLY}; Q=$?
echo "$(date '+%F %T')  assembly done (monthly exit ${M}, quarterly exit ${Q}; 0 = every period of ${YEAR} assembled)"

# The checks run on whatever of the year is assembled, also if some periods were refused.
echo "$(date '+%F %T')  checking ${YEAR}"
python -u panel_pipeline/checks/check_panel.py --run "${RUN}" --year "${YEAR}"
C=$?
echo "$(date '+%F %T')  done: assembly ${M}/${Q}, checks ${C} (all 0 = everything fine)"
[ ${M} -eq 0 ] && [ ${Q} -eq 0 ] && [ ${C} -eq 0 ]
