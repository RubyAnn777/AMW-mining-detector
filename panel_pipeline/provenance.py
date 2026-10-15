"""Where a result came from: the machine, the code, the packages, the input files.

plan.py records this once per run. detect.py records it again for every period
it scores, because scoring can happen later, on another machine, with newer code.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from config import REPO, Config

PACKAGES = ("descarteslabs", "earthengine-api", "tensorflow", "geopandas",
            "shapely", "numpy", "pandas", "pyarrow")

# DEVIATION from AMW: AMW records how detections were made only in file names
# (model, threshold, dates) and writes a settings file only for SAM2 masks
# (MaskConfig.write_config, inference_engine.py:314). We record every setting
# once per run, plus the code version and a fingerprint of each input file,
# so any number in the panel can be traced back to exactly what made it.


def _git(*args: str) -> str | None:
    """Ask git about the code in use; None if git is missing or this isn't a checkout."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                             text=True, check=True)   # as scripts/provenance_report.py:31
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.rstrip()       # drop the final newline, keep leading spaces


def _uncommitted(cfg: Config) -> list[str] | None:
    """Files changed since the last commit, except the config (the run keeps a copy)."""
    status = _git("status", "--porcelain")      # one line per file: "XY path"
    if status is None:
        return None
    config = cfg.source.relative_to(REPO).as_posix() if cfg.source.is_relative_to(REPO) else None
    return [line[3:] for line in status.splitlines() if line[3:] != config]


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    """A file's fingerprint: it changes if a single byte of the file changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    """A path as written in the records: relative to the repository when inside it."""
    return str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)


def environment(cfg: Config) -> dict:
    """The machine, code and package versions in use right now."""
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_files": _uncommitted(cfg),
        "packages": {p: _version(p) for p in PACKAGES},
    }


def fingerprints(cfg: Config) -> dict[str, str]:
    """The fingerprint of every input file: the model and the boundary files."""
    inputs = [cfg.inference.model_path, cfg.region.andes_boundary,
              *cfg.region.subregion_files.values()]
    if cfg.region.clip_to:
        inputs.append(cfg.region.clip_to)
    return {repo_path(p): sha256(p) for p in inputs}
