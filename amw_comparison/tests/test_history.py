"""Tests for history.py: tile numbers, keys, rings. No downloads.

    python -m unittest discover -s amw_comparison/tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(1, str(HERE.parent.parent / "panel_pipeline" / "lib"))

import compare  # noqa: E402
import history  # noqa: E402

SIZE = 5520.0


def number(zone, col, row):
    return (zone * 1000 + col + 500) * 10_000 + row + 5000


class TestHistory(unittest.TestCase):
    def test_split_undoes_key_number(self):
        key = "552:12:10.0:21:60:-100"
        z, c, r = history.split_tile(compare.key_number(key))
        self.assertEqual((int(z), int(c), int(r)), (21, 60, -100))

    def test_square_is_the_tile(self):
        """Our square for a tile is DLTile's own (without the pad)."""
        from descarteslabs.geo import DLTile
        tile = DLTile.from_key("552:0:10.0:21:60:-100")
        ours = history.tile_squares([number(21, 60, -100)], SIZE)[0]
        self.assertLess(ours.symmetric_difference(tile.geometry).area / tile.geometry.area, 1e-3)

    def test_key_int_is_one_to_one_at_5_decimals(self):
        x = np.array([-60.12345, -60.12346, -60.12345])
        y = np.array([-3.5, -3.5, -3.50001])
        self.assertEqual(len(set(history.key_int(x, y))), 3)

    def test_rings_in_one_zone(self):
        cols, rows = np.meshgrid(np.arange(50, 59), np.arange(-104, -95))
        tiles = pd.DataFrame({"tile": number(21, cols.ravel(), rows.ravel())})
        centre = (tiles["tile"] == number(21, 54, -100)).to_numpy()
        ring = pd.Series(history.rings(tiles, centre, SIZE), index=tiles["tile"])
        z, c, r = history.split_tile(tiles["tile"])
        expected = np.minimum(np.maximum(abs(c - 54), abs(r + 100)), history.FAR)
        np.testing.assert_array_equal(ring.to_numpy(), expected)

    def test_rings_cross_a_zone_seam(self):
        """A tile at the east edge of zone 20 has neighbours in zone 21."""
        east = history.tile_squares([number(20, 59, -201)], SIZE)[0]
        cand = [number(21, c, r) for c in range(-62, -57) for r in range(-204, -198)]
        touching = [n for n, s in zip(cand, history.tile_squares(cand, SIZE)) if s.distance(east) < 1e-6]
        self.assertTrue(touching)
        tiles = pd.DataFrame({"tile": [number(20, 59, -201)] + cand})
        ring = history.rings(tiles, np.r_[True, np.zeros(len(cand), bool)], SIZE)
        self.assertTrue((ring[1:][np.isin(cand, touching)] <= 1).all())


if __name__ == "__main__":
    unittest.main()
