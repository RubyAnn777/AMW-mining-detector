"""TensorFlow defaults on macOS to avoid kernel crashes with Metal / tf.data."""

# >> FILE OVERVIEW
# >> Tiny macOS workaround: on Apple Silicon, TensorFlow's Metal GPU plugin and
# >> tf.data can crash, so on Darwin (macOS) we force slower-but-safe eager mode.
# >> Upstream file (Amazon Mining Watch), byte-identical. Called at import time by
# >> inference_engine.py and model_library.py. Does nothing on Linux (our VMs).
from __future__ import annotations

import platform

# >> Module flag so the settings are applied at most once per process.
_applied = False


# >> On macOS only: switch TensorFlow to eager (line-by-line) execution and turn on
# >> tf.data debug mode. Idempotent (safe to call many times); no-op elsewhere.
# >> Why: avoids kernel crashes with Metal / tf.data; costs speed on a Mac.
def apply_darwin_tf_compat() -> None:
    """Enable eager execution and tf.data debug mode on Darwin (idempotent).

    Call after ``import tensorflow`` and before building datasets or training.
    Safe no-op on non-macOS.
    """
    global _applied
    # >> platform.system() is 'Darwin' on macOS, 'Linux' on our servers.
    if _applied or platform.system() != "Darwin":
        return
    import tensorflow as tf

    # >> Run tf.functions eagerly (no graph compilation).
    tf.config.run_functions_eagerly(True)
    # >> tf.data debug mode: pipelines run in plain Python order (deterministic, slower).
    tf.data.experimental.enable_debug_mode()
    _applied = True
