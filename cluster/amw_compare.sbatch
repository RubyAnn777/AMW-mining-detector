#!/bin/bash
# Compare one year of a run with AMW's published quarters (amw_comparison/compare.py).
#
#     sbatch cluster/amw_compare.sbatch <run folder> <year>
#
# Needs AMW's files in amw_comparison/data/amw/ on Goethe (copied from the laptop, not in
# git), e.g. amazon_basin_Q126_t0.4.geojson for 2026Q1. Reads detect's output directly,
# not the panel, so it can run before, after or during assembly.
# Writes <run>/amw_comparison/: chips_<quarter>.parquet per quarter, and inputs.csv,
# joins.csv, amw_files.csv, where a later year adds its rows and keeps the earlier ones.
# Then look at it in notebooks/amw_comparison.py (cluster/view.sbatch), after cluster/summaries.sbatch.
#
# Measured: ~2 min and 6.5 GB per subregion and quarter (README of amw_comparison/).
#
#SBATCH --job-name=amw-compare
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=compare-%j.out

USAGE="usage: sbatch cluster/amw_compare.sbatch <run folder> <year>"
RUN="${1:?${USAGE}}"
YEAR="${2:?${USAGE}}"
source cluster/goethe_env.sh               # run sbatch from the repository root
echo "$(date '+%F %T')  comparing ${YEAR} of ${RUN} with AMW, code $(git rev-parse --short HEAD)"
ls -l amw_comparison/data/amw/ 2>/dev/null || echo "no folder amw_comparison/data/amw/ (copy AMW's files first)"
python -u amw_comparison/compare.py --run "${RUN}" --year "${YEAR}"
S=$?
echo "$(date '+%F %T')  done (exit ${S}; 0 = fine)"
exit ${S}
