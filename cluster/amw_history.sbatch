#!/bin/bash
# Which tiles has AMW ever found mining in, and how many of our 2025/2026 detections lie in
# or next to them (amw_comparison/history.py, then notebooks/amw_history.py as a web page).
#
#     sbatch cluster/amw_history.sbatch <run folder>
#     sbatch --dependency=afterok:<compare job> cluster/amw_history.sbatch <run folder>
#                                       start only once amw_compare.sbatch has finished
#
# Needs AMW's cumulative record in amw_comparison/data/amw/amazon_basin_detections.geojson
# on Goethe (copied from the laptop, not in git), and for the page, the compared quarters
# (chips_<quarter>.parquet from amw_compare.sbatch).
# Writes <run>/amw_comparison/history_tiles.parquet, history_tiles.geojson,
# history_summary.json, and the page <run>/reports/amw_history.html.
#
# Estimate: a few minutes (one pass over plan/chips.parquet, then the rings).
#
#SBATCH --job-name=amw-history
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=history-%j.out

USAGE="usage: sbatch cluster/amw_history.sbatch <run folder>"
RUN="${1:?${USAGE}}"
RUN="${RUN%/}"
source cluster/goethe_env.sh               # run sbatch from the repository root
echo "$(date '+%F %T')  AMW history tiles for ${RUN}, code $(git rev-parse --short HEAD)"
ls -l amw_comparison/data/amw/amazon_basin_detections.geojson \
    || echo "missing: copy amazon_basin_detections.geojson into amw_comparison/data/amw/ first"
python -u amw_comparison/history.py --run "${RUN}"
S=$?
if [ ${S} -eq 0 ]; then
    mkdir -p "${RUN}/reports"
    marimo export html notebooks/amw_history.py -o "${RUN}/reports/amw_history.html" -- --run "${RUN}"
    S=$?
    chmod -R g+rX "${RUN}/reports" 2>/dev/null
    ls -l "${RUN}/reports/amw_history.html"
fi
echo "$(date '+%F %T')  done (exit ${S}; 0 = fine)"
exit ${S}
