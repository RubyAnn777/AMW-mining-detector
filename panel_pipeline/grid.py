"""The panel's space axis: AMW's tiles and chips, and one whole-number chip ID.

periods.py says WHEN a row is; this file says WHERE. Tiles and chip squares
come from AMW's own functions, so a chip here is exactly a chip AMW scores.
"""
from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np

from config import REPO

# AMW's files import each other by bare name with core/ on the search path,
# as scripts/geo_filter.py:12 does. We APPEND core/ rather than put it first:
# AMW also has a core/periods.py, and `import periods` must find ours.
sys.path.append(str(REPO / "core"))
from postprocess import centroid_key           # noqa: E402  (AMW code)
from tile_utils import create_tiles, cut_chips  # noqa: E402  (AMW code)

CHIP_PX = 48          # AMW InferenceConfig.geo_chip_size (inference_engine.py:161): the model's input
RESOLUTION_M = 10     # AMW create_tiles default (tile_utils.py:102), used by inference_pipeline.py:61