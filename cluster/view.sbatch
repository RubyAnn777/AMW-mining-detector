#!/bin/bash
# Open the notebooks in your browser, for coauthors and for Ruby. Starts a marimo server
# on a compute node; you reach it through VS Code's port forwarding (or an ssh tunnel).
#
#     sbatch /work/goldmine/grambauer/amw-view/cluster/view.sbatch            the run changed last
#     sbatch /work/goldmine/grambauer/amw-view/cluster/view.sbatch <run>      another run
#     sbatch --time=06:00:00 --partition=general1 .../view.sbatch            longer than 2 hours
#
# Then: cat ~/marimo-view-<jobid>.out (after a few seconds). It prints the node and the
# address to open. The server stops when the job's time is up, or with scancel <jobid>.
#
# Read-only for everyone but the run's owner: the notebooks open as apps (no code shown,
# nothing editable), and they read Ruby's run folder without writing to it.
#
#SBATCH --job-name=amw-view
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

SHARED="${AMW_SHARED:-/work/goldmine/grambauer}"           # Ruby's folder on /work
VIEW_REPO="${VIEW_REPO:-${SHARED}/amw-view}"               # the read-only copy of the repository
VENV="${VENV:-${SHARED}/venvs/amw-panel}"                  # the one shared Python environment
RUN="${1:-$(ls -1dt "${SHARED}"/runs/*/ 2>/dev/null | head -1)}"   # default: the run changed last
RUN="${RUN%/}"
LOG="${HOME}/marimo-view-${SLURM_JOB_ID}.out"

umask 077                                                  # the log holds your access token: only you can read it
exec > "${LOG}" 2>&1

[ -d "${RUN}/run" ] || { echo "VIEW ERROR: ${RUN} is not a run folder"; exit 2; }
source "${VENV}/bin/activate" || { echo "VIEW ERROR: no environment at ${VENV}"; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=8

PORT=$(( 20000 + $(id -u) % 20000 ))                       # the same port for you every time
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(16))")
NODE=$(hostname -s)
cat <<MSG
The notebooks are starting on ${NODE} for run ${RUN}.

1. In VS Code (connected to Goethe), open the PORTS tab next to TERMINAL,
   click "Forward a Port" and type:   ${NODE}:${PORT}
   (Without VS Code, run this on your laptop instead:
    ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@goethe.hhlr-gu.de )
2. Open in your browser:
   http://localhost:${PORT}/?access_token=${TOKEN}

The server runs until $(squeue -h -j "${SLURM_JOB_ID}" -o %e 2>/dev/null | tr T ' ') (scancel ${SLURM_JOB_ID} stops it sooner).
Keep this address to yourself: anyone on Goethe with it could open your session.

MSG
cd "${VIEW_REPO}"
exec marimo run notebooks/ --host 0.0.0.0 --port "${PORT}" --headless \
     --token --token-password "${TOKEN}" --no-sandbox --session-ttl 600 -- --run "${RUN}"
