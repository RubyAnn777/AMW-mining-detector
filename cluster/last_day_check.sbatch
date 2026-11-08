#!/bin/bash
# Test whether the quarter's last day explains every gap with AMW (amw_comparison/last_day_check.py).
#
#     sbatch cluster/last_day_check.sbatch <run folder> <year> [tiles per subregion]
#     sbatch cluster/last_day_check.sbatch $RUN 2026            # 4 tiles per subregion (default)
#     sbatch cluster/last_day_check.sbatch $RUN 2026 10         # more tiles, a stronger test
#
# Needs compare.py's results for the year first (cluster/amw_compare.sbatch). For every
# quarter of the year it writes <run>/amw_comparison/last_day_check/<quarter>/: a table
# per subregion, the disagreeing chips outside the last day's scenes, and the re-scoring
# of sample tiles three ways (AMW's code + AMW's dates, our code + our dates, AMW's code +
# our dates). summary.txt per quarter says what passed; this job's log prints them all.
#
# Earth Engine: per quarter, one small request for the last day's scene outlines and 3
# tile downloads per re-scored tile (6 subregions x (4 + 1 control) x 3 = 90 by default,
# about 1% of a month's quota per quarter at ~145 EECU-seconds a download).
#
#SBATCH --job-name=amw-lastday
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=lastday-%j.out

USAGE="usage: sbatch cluster/last_day_check.sbatch <run folder> <year> [tiles per subregion]"
RUN="${1:?${USAGE}}"
YEAR="${2:?${USAGE}}"
TILES="${3:-4}"
export TF_THREADS=40                       # one process: give TensorFlow the whole node
source cluster/goethe_env.sh               # run sbatch from the repository root
echo "$(date '+%F %T')  last-day check of ${YEAR}, ${RUN}, ${TILES} tiles per subregion, code $(git rev-parse --short HEAD)"
python -u amw_comparison/last_day_check.py --run "${RUN}" --year "${YEAR}" --tiles "${TILES}"
S=$?
echo "$(date '+%F %T')  done (exit ${S}; 0 = ran through; whether the tests passed is in the summaries above)"
exit ${S}
