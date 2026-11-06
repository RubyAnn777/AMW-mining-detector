#!/bin/bash
# Run a notebook from top to bottom on a compute node and save it as a web page (HTML).
#
#     sbatch cluster/notebook.sbatch <notebook> <run folder> [SETTING=value ...]
#
#     sbatch cluster/notebook.sbatch panel_pipeline/checks/panel_overview.ipynb $RUN
#     sbatch cluster/notebook.sbatch panel_pipeline/checks/panel_overview.ipynb $RUN MAP_PERIOD=2026Q2
#     sbatch cluster/notebook.sbatch amw_comparison/compare.ipynb $RUN YEAR=2026
#     sbatch cluster/notebook.sbatch panel_pipeline/checks/explore_runs.ipynb $RUN FOCUS_QUARTER=2026Q3
#     sbatch cluster/notebook.sbatch panel_pipeline/checks/performance.ipynb $RUN
#
# The settings are the ones in the notebook's first cell that read os.environ (RUN,
# SUBREGIONS, MAP_PERIOD, YEAR, QUARTERS, ...); anything not given keeps the notebook's
# own value, except SUBREGIONS, which is "all" here (the whole region's panel).
# Result: <run>/reports/<notebook>_<date>_<time>.html, with every table and figure, and
# the same as .ipynb. Copy the .html to the laptop and open it in a browser (cheat sheet).
# If a cell fails, the page is still written, with the error under that cell, and the
# log names it.
#
#SBATCH --job-name=amw-notebook
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=notebook-%j.out

USAGE="usage: sbatch cluster/notebook.sbatch <notebook.ipynb> <run folder> [SETTING=value ...]"
NOTEBOOK="${1:?${USAGE}}"
RUN="${2:?${USAGE}}"
shift 2
[ -f "${NOTEBOOK}" ] || { echo "NOTEBOOK ERROR: no file ${NOTEBOOK} (run sbatch from the repository root)"; exit 2; }
NOTEBOOK="$(realpath "${NOTEBOOK}")"
source cluster/goethe_env.sh               # run sbatch from the repository root

export RUN
export SUBREGIONS="${SUBREGIONS:-all}"
mkdir -p "${RUN}/reports"
export EXPORT_DIR="${EXPORT_DIR:-${RUN}/reports/panel_tables}"   # panel_overview's CSV tables
for SETTING in "$@"; do                    # MAP_PERIOD=2026Q2 -> the notebook sees it
    [[ "${SETTING}" == *=* ]] || { echo "NOTEBOOK ERROR: '${SETTING}' is not SETTING=value"; exit 2; }
    export "${SETTING}"
done

NAME="$(basename "${NOTEBOOK}" .ipynb)_$(date '+%Y-%m-%d_%H%M')"
OUT="${RUN}/reports"
echo "$(date '+%F %T')  running ${NOTEBOOK} for ${RUN} $*, code $(git rev-parse --short HEAD)"
# Run every cell once, keeping going past a failing one so the page shows where it failed;
# then save the run notebook (.ipynb, opens in Jupyter) and the page (.html, any browser).
jupyter nbconvert --to notebook --execute --allow-errors "${NOTEBOOK}" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "${OUT}" --output "${NAME}" || exit 1
jupyter nbconvert --to html "${OUT}/${NAME}.ipynb" --output-dir "${OUT}" --output "${NAME}" || exit 1
python - "${OUT}/${NAME}.ipynb" <<'PY'
import json, sys
cells = json.load(open(sys.argv[1]))["cells"]
errors = [(i, o) for i, c in enumerate(cells) if c["cell_type"] == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
for i, o in errors:
    print(f"CELL {i} FAILED: {o['ename']}: {o['evalue']}")
sys.exit(1 if errors else 0)
PY
S=$?
[ ${S} -eq 0 ] && echo "$(date '+%F %T')  done, no errors: ${OUT}/${NAME}.html" \
               || echo "$(date '+%F %T')  done WITH ERRORS (above); see them in ${OUT}/${NAME}.html"
exit ${S}
