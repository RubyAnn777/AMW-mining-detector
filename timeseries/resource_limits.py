# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
resource_limits.py — cap CPU parallelism before TensorFlow initialises.

WHY THIS EXISTS
---------------
Audit of 2026-08-04 (docs/RUN2_SETUP_AND_AUDIT.md §3) found NO CPU cap, memory cap,
process-priority setting or thermal safeguard anywhere in this codebase. The dominant
sustained load is ``model.predict`` (core/inference_engine.py:620), and TensorFlow
defaults to using every available core. Measured per tile: infer ≈ 7.5 s vs fetch ≈ 2.4 s
(data/timeseries/tiles/tile_stats_m2025-01.csv), so inference is where the heat is.

WHAT IT DOES
------------
1. Sets the BLAS/OpenMP thread environment variables. These are read at library import
   time, so this must run BEFORE ``import tensorflow`` / ``numpy`` anywhere in the
   process — hence the call sits at the top of run_trials.py, before ``import panel``.
2. Sets TensorFlow's intra-/inter-op thread pools via the public API. Env vars alone are
   not sufficient for TF; the documented control is
   ``tf.config.threading.set_intra_op_parallelism_threads``. This must be called before
   any op executes, otherwise TF raises RuntimeError — we catch and warn rather than
   abort, because a warning is preferable to killing a long run.
3. Sets ``torch.set_num_threads`` if torch is importable (it is imported by
   inference_engine even on the ``embedding_strategy='none'`` path).

WHAT IT DOES NOT DO
-------------------
It does not monitor temperature and it cannot throttle dynamically. Process priority is
set by the launcher (timeseries/run_staged.sh), not here, because it must be applied to
the process before Python starts.

CONFIGURATION
-------------
``AMW_THREADS`` environment variable (default 4). Set to 0 to disable all capping and
restore stock behaviour.
"""
# >> FILE OVERVIEW: fork-original (not upstream). Stops the run from using every CPU core.
# >> Caps threads for numpy/BLAS (env vars), TensorFlow (intra/inter-op pools) and torch.
# >> Called once at the top of run_trials.py, before TensorFlow is imported.
# >> Budget from env AMW_THREADS (default 4; 0 = no cap). It does not watch temperature;
# >> run_staged.sh handles priority and cooldowns.
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# >> Default thread budget when AMW_THREADS is unset: 4 threads.
_DEFAULT_THREADS = 4

# >> These env vars are read once, when the library is imported; setting them later does
# >> nothing, hence "before numpy/TF import".
# Read by numpy/scipy BLAS backends and OpenMP at import time.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",      # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
)

# >> Module-level flag so repeated calls do nothing (idempotent).
_applied = False


# >> What: decides how many CPU threads to allow.
# >> Priority: explicit argument > AMW_THREADS env var > default 4. Returns an int.
# >> A non-integer AMW_THREADS logs a warning and falls back to 4.
def resolve_threads(threads: Optional[int] = None) -> int:
    """Thread budget: explicit argument > ``AMW_THREADS`` env > default 4."""
    if threads is not None:
        return int(threads)
    raw = os.environ.get("AMW_THREADS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            log.warning("AMW_THREADS=%r is not an integer; using %d",
                        raw, _DEFAULT_THREADS)
    return _DEFAULT_THREADS


# >> What: applies the thread cap everywhere it matters; returns the budget n.
# >> Steps: 1) OMP / OpenBLAS / MKL / Accelerate / NumExpr env vars = n (only if not set);
# >> 2) TF intra_op = n, inter_op = max(1, n // 2); 3) torch.set_num_threads(n).
# >> Must run before TF/numpy import: env vars are read at import, and TF pools can only
# >> be set before the first TF op (else RuntimeError, caught and logged as a warning).
# >> Called by run_trials.py at import time. Gotcha: logging is not configured yet then,
# >> so these INFO messages usually do not appear; run_trials.py logs describe() instead.
def apply_resource_limits(threads: Optional[int] = None) -> int:
    """Cap CPU parallelism. Idempotent. Returns the thread budget applied.

    ``threads <= 0`` disables capping entirely (stock TensorFlow behaviour).
    Must be called before TensorFlow/numpy are imported for the env vars to bite.
    """
    global _applied
    if _applied:
        return resolve_threads(threads)

    n = resolve_threads(threads)
    # >> n <= 0: no caps at all (stock behaviour, all cores).
    if n <= 0:
        log.info("resource_limits: capping DISABLED (AMW_THREADS<=0); "
                 "TensorFlow will use all cores")
        _applied = True
        return n

    # --- 1. env vars (must precede numpy/TF import) -------------------------
    # >> setdefault: a value already exported in the shell wins over n.
    for var in _THREAD_ENV_VARS:
        # setdefault: an explicit value from the shell always wins
        os.environ.setdefault(var, str(n))

    # --- 2. TensorFlow thread pools ----------------------------------------
    # >> Example with n = 4: each conv/matmul may use 4 threads; 2 independent ops at
    # >> once.
    # intra_op = threads within a single op (the matmuls/convs — the hot path)
    # inter_op = independent ops in parallel; half, min 1
    inter = max(1, n // 2)
    try:
        # >> Importing TF here is deliberate: TF starts only after the env vars above are
        # >> set.
        import tensorflow as tf  # noqa: PLC0415 (deliberately late)

        # >> TF thread pools: intra-op = inside one op, inter-op = across independent ops.
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        log.info("resource_limits: TF intra_op=%d inter_op=%d", n, inter)
    except RuntimeError as e:
        # raised if TF context is already initialised — too late to change
        log.warning("resource_limits: could not set TF thread pools (%s). "
                    "Call apply_resource_limits() earlier, before any TF op.", e)
    except ImportError:
        log.warning("resource_limits: tensorflow not importable; "
                    "env vars set, TF pools unchanged")

    # --- 3. torch ----------------------------------------------------------
    try:
        # >> torch is imported by inference_engine even on the CNN path, so cap it too.
        import torch  # noqa: PLC0415

        torch.set_num_threads(n)
    except Exception as e:  # torch optional on the CNN-only path
        log.debug("resource_limits: torch thread cap skipped (%s)", e)

    log.info("resource_limits: CPU thread budget = %d (%s)",
             n, ", ".join(f"{v}={os.environ[v]}" for v in _THREAD_ENV_VARS
                          if v in os.environ))
    _applied = True
    return n


# >> What: one-line text of the effective thread settings, for the run log.
# >> Out: e.g. "OMP_NUM_THREADS=4 ... tf_intra=4 tf_inter=2"; missing env vars show
# >> 'unset'.
# >> Called by run_trials.py:main right after logging is set up.
def describe() -> str:
    """One-line summary of the effective limits, for logging."""
    parts = [f"{v}={os.environ.get(v, 'unset')}" for v in _THREAD_ENV_VARS]
    try:
        import tensorflow as tf

        parts.append(
            f"tf_intra={tf.config.threading.get_intra_op_parallelism_threads()}")
        parts.append(
            f"tf_inter={tf.config.threading.get_inter_op_parallelism_threads()}")
    except Exception:
        pass
    return " ".join(parts)
