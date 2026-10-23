# Settings for the panel pipeline on Goethe-NHR. Source it in every shell and job:
#     source cluster/goethe_env.sh
# Fill in the three lines marked CHANGE once. Nothing here is a secret: the key file's
# *path* is public, the file itself is not (chmod 600, never in git).

GOETHE_GROUP="goldmine"                                    # your group: `ls /work` shows it
GOETHE_USER="${USER}"                                    # your login name
GOETHE_WORK="/work/${GOETHE_GROUP}/${GOETHE_USER}"

# The Python environment (made once: README B2) and uv's folders, all on /work.
source "${GOETHE_WORK}/venvs/amw-panel/bin/activate"
export UV_CACHE_DIR="${GOETHE_WORK}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${GOETHE_WORK}/python"

# Earth Engine: the service account's key file. AMW's gee.py reads this variable
# (core/gee.py:69) and logs in as that account; without it, your own login is used.
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.secrets/ee-amw-panel.json"   # CHANGE if elsewhere

# TensorFlow threads per detect.py process. A Goethe node has 40 cores and is yours
# alone, so without this every process would start 40 threads and several processes
# would fight over the cores. Set it to about 40 / (tasks per node); the sbatch script
# overrides it from its arguments.
export TF_THREADS="${TF_THREADS:-10}"
export TF_NUM_INTRAOP_THREADS="${TF_THREADS}"     # threads inside one operation (the model)
export TF_NUM_INTEROP_THREADS=1                   # operations run in parallel: one is enough
export OMP_NUM_THREADS="${TF_THREADS}"            # the maths libraries TensorFlow and NumPy use
export TF_CPP_MIN_LOG_LEVEL=1                     # hide TensorFlow's start-up chatter

cd "${HOME}/AMW-mining-detector"                  # CHANGE if the repository is elsewhere
