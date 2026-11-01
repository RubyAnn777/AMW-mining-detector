"""The panel's space axis: AMW's tiles and chips, and one whole-number chip ID.

Tiles and chip squares come from AMW's functions
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

# ----------------------------------------------------------------- tiles

def read_shape(path):
    """A boundary file as one shape, the way AMW reads a region."""
    return gpd.read_file(path).union_all()        # AMW: inference_pipeline.py:58


def tiles_for(shape, imagery) -> list:
    """AMW's tiles over a shape: the global DLTile grid, unchanged."""
    if shape.is_empty:
        return []
    return create_tiles(shape, imagery.tilesize, imagery.pad, RESOLUTION_M)  # AMW: inference_pipeline.py:61


# ----------------------------------------------------------------- chip IDs

# DEVIATION from AMW: AMW has no chip ID. It recognises "the same chip in
# another period" by rounding the chip's centre to 5 decimals (postprocess.py:113)
# and its own comments show that join is fragile (postprocess.py:62-81: the
# match count moves by 146,485 with the rounding). Our chip_id is computed from
# the tile and the chip's position, so it is exact and the same in every period.
#
# Layout, read digit by digit:  zone | tile column + 500 | tile row + 5000 | chip
#                       e.g.    19   |       485         |      4737       | 417
# Whole numbers, because the panel repeats chip_id on every row (about 20
# billion rows); text IDs would cost several times the memory.

def chip_ids(tile, n_chips: int) -> np.ndarray:
    """IDs of a tile's chips, in AMW's chip order (cut_chips, tile_utils.py:213)."""
    if not (1 <= tile.zone <= 60 and -500 <= tile.ti < 500
            and -5000 <= tile.tj < 5000 and n_chips <= 1000):
        raise ValueError(f"tile {tile.key} does not fit the chip_id layout")
    base = ((tile.zone * 1000 + tile.ti + 500) * 10_000 + tile.tj + 5000) * 1000
    return base + np.arange(n_chips, dtype=np.int64)


def split_chip_id(chip_id: int) -> tuple[int, int, int, int]:
    """chip_id -> (zone, tile column, tile row, chip index). Undoes chip_ids."""
    rest, chip = divmod(int(chip_id), 1000)
    rest, tj = divmod(rest, 10_000)
    zone, ti = divmod(rest, 1000)
    return zone, ti - 500, tj - 5000, chip

# ----------------------------------------------------------------- chips

def chip_table(tile, stride: int) -> gpd.GeoDataFrame:
    """Every chip AMW's model scores in this tile, with its square and keys."""
    # cut_chips wants pixels, but the chip squares depend only on the tile's
    # bounds, so a blank one-band image gives AMW's exact squares without imagery.
    side = tile.tilesize + 2 * tile.pad
    blank = np.zeros((side, side, 1), dtype=np.uint8)
    _, chips = cut_chips(blank, tile.geometry.bounds, CHIP_PX, stride,
                         crs="epsg:4326")          # AMW: inference_engine.py:949
    key_x, key_y = centroid_key(chips)             # AMW's location key, kept for comparison
    b = chips.geometry.bounds
    return chips.assign(
        chip_id=chip_ids(tile, len(chips)),
        tile_key=tile.key,
        chip_index=np.arange(len(chips), dtype=np.int16),
        minx=b.minx, miny=b.miny, maxx=b.maxx, maxy=b.maxy,
        amw_key_x=key_x, amw_key_y=key_y,
    )

