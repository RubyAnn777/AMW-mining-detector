#!/bin/bash
# Store the notebooks' summaries for a run, so they open in seconds for everyone.
#
#     sbatch cluster/summaries.sbatch <run folder>
#     sbatch cluster/summaries.sbatch <run folder> 2026Q2      also that quarter's example rows and distributions
#
# First builds the analysis tables for notebooks 1 to 4 (notebooks/build_tables.py, into
# <run>/analysis/; only new or changed periods are read). Then runs panel_overview and
# amw_comparison once from top to bottom, as plain Python scripts. Whatever a notebook
# reads from the whole run (the checks, the distributions, ...) is stored in <run>/summaries/, which
# your coauthors can read but not write. Without it, those sections of panel_overview say
# "Not stored yet" in the browser: on a full run they take too long to compute while a
# page opens.
#
# Submit it after assemble.sbatch and amw_compare.sbatch (cheat sheet section 9). Safe to
# submit again: a summary is only recomputed when the run changed since it was stored.
#
#SBATCH --job-name=amw-summaries
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=summaries-%j.out

USAGE="usage: sbatch cluster/summaries.sbatch <run folder> [quarter]"
RUN="${1:?${USAGE}}"
QUARTER="${2:-}"
source cluster/goethe_env.sh               # run sbatch from the repository root
umask 027                                  # the summaries: readable by the group, writable only by you
echo "$(date '+%F %T')  summaries for ${RUN}, code $(git rev-parse --short HEAD)"
S=0
START=$(date +%s)
if python notebooks/build_tables.py --run "${RUN}"; then
    echo "$(date '+%F %T')  analysis tables: done ($(( ($(date +%s) - START) / 60 )) min)"
else
    echo "$(date '+%F %T')  analysis tables: FAILED (the error is above)"; S=1
fi
chmod -R g+rX "${RUN}/analysis" 2>/dev/null
for NB in panel_overview amw_comparison; do
    ARGS=(--run "${RUN}")
    [ -n "${QUARTER}" ] && [ "${NB}" = panel_overview ] && ARGS+=(--period "${QUARTER}")
    START=$(date +%s)
    if python "notebooks/${NB}.py" "${ARGS[@]}"; then
        echo "$(date '+%F %T')  ${NB}: done ($(( ($(date +%s) - START) / 60 )) min)"
    else
        echo "$(date '+%F %T')  ${NB}: FAILED (the error is above)"; S=1
    fi
done
chmod -R g+rX "${RUN}/summaries" 2>/dev/null
ls -l "${RUN}/summaries"
echo "$(date '+%F %T')  done (exit ${S}; 0 = fine)"
exit ${S}
