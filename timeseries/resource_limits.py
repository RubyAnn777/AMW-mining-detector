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
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_THREADS = 4

# Read by numpy/scipy BLAS backends and OpenMP at import time.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",      # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
)

_applied = False


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


def apply_resource_limits(threads: Optional[int] = None) -> int:
    """Cap CPU parallelism. Idempotent. Returns the thread budget applied.

    ``threads <= 0`` disables capping entirely (stock TensorFlow behaviour).
    Must be called before TensorFlow/numpy are imported for the env vars to bite.
    """
    global _applied
    if _applied:
        return resolve_threads(threads)

    n = resolve_threads(threads)
    if n <= 0:
        log.info("resource_limits: capping DISABLED (AMW_THREADS<=0); "
                 "TensorFlow will use all cores")
        _applied = True
        return n

    # --- 1. env vars (must precede numpy/TF import) -------------------------
    for var in _THREAD_ENV_VARS:
        # setdefault: an explicit value from the shell always wins
        os.environ.setdefault(var, str(n))

    # --- 2. TensorFlow thread pools ----------------------------------------
    # intra_op = threads within a single op (the matmuls/convs — the hot path)
    # inter_op = independent ops in parallel; half, min 1
    inter = max(1, n // 2)
    try:
        import tensorflow as tf  # noqa: PLC0415 (deliberately late)

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
        import torch  # noqa: PLC0415

        torch.set_num_threads(n)
    except Exception as e:  # torch optional on the CNN-only path
        log.debug("resource_limits: torch thread cap skipped (%s)", e)

    log.info("resource_limits: CPU thread budget = %d (%s)",
             n, ", ".join(f"{v}={os.environ[v]}" for v in _THREAD_ENV_VARS
                          if v in os.environ))
    _applied = True
    return n


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
