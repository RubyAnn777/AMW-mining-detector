from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from periods import FREQUENCIES, Period, period_range

PIPELINE_DIR = Path(__file__).resolve().parent     # .../panel_pipeline
REPO = PIPELINE_DIR.parent                         # the repository root
DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"


@dataclass(frozen=True)
class Region:
    subregion_files: dict[int, Path]     # 1 -> .../Amazon_ACA_1.geojson, ...
    andes_boundary: Path

@dataclass(frozen=True)
class Imagery:
    collection: str
    tilesize: int
    pad: int
    clear_threshold: float
    max_workers: int


@dataclass(frozen=True)
class Inference:
    model_path: Path
    stride_ratio: int
    tries: int
    batch_tiles: int


@dataclass(frozen=True)
class DualThreshold:
    t_main: float
    t_iso: float


@dataclass(frozen=True)
class Thresholds:
    raw: float
    andes: float
    loose: DualThreshold
    stringent: DualThreshold
    neighbours_k: int
    isolation_km: float


@dataclass(frozen=True)
class Persistence:
    k: int
    window_months: int


@dataclass(frozen=True)
class Config:
    source: Path                          # the config.yaml these came from
    run_name: str
    output_root: Path
    periods: dict[str, list[Period]]      # "monthly" -> 105 periods, ...
    region: Region
    imagery: Imagery
    inference: Inference
    thresholds: Thresholds
    persistence: Persistence