"""Tests for compare.py and scene_dates.py. No Earth Engine, no downloads.

    python -m unittest discover -s amw_comparison/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                            # compare.py, scene_dates.py
sys.path.insert(1, str(HERE.parent.parent / "panel_pipeline"))  # grid.py, for real chips

import compare  # noqa: E402

TILE = "552:12:10.0:21:60:-100"   # a subregion 3 tile; DLTile.from_key needs no network


class TestTags(unittest.TestCase):
    def test_amw_tag(self):
        self.assertEqual(compare.amw_tag("2026Q3"), "Q326")
        self.assertEqual(compare.amw_file("2026Q1").name, "amazon_basin_Q126_t0.4.geojson")


class TestJoinOnRealChips(unittest.TestCase):
    """The join must find every chip again after a round trip through an AMW-style file."""

    @classmethod
    def setUpClass(cls):
        from descarteslabs.geo import DLTile
        from grid import chip_table
        cls.tile = DLTile.from_key(TILE)
        cls.squares = chip_table(cls.tile, stride=24)
        cls.chips = pd.DataFrame(cls.squares.drop(columns="geometry")).assign(
            tile=lambda c: c["chip_id"] // 1000)

    def amw_rows(self, squares, confidence) -> pd.DataFrame:
        # As AMW writes it: squares in EPSG:4326, 6 decimals (concatenate.py)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "amw.geojson"
            squares[["geometry"]].assign(confidence=confidence).to_file(
                path, driver="GeoJSON", COORDINATE_PRECISION=6)
            return compare.read_amw(path)

    def test_every_chip_matches_exactly_once(self):
        joined, unmatched = compare.match(self.chips, self.amw_rows(self.squares, 0.9))
        self.assertEqual(len(self.chips), 529)
        self.assertEqual(len(unmatched), 0)
        self.assertTrue(joined["amw_confidence"].notna().all())

    def test_a_subset_lands_on_the_right_chips(self):
        amw = self.amw_rows(self.squares.iloc[[0, 1, 23, 528]], [0.5, 0.6, 0.7, 0.8])
        joined, _ = compare.match(self.chips, amw)
        got = joined.dropna(subset=["amw_confidence"]).set_index("chip_index")["amw_confidence"]
        self.assertEqual(got.to_dict(), {0: 0.5, 1: 0.6, 23: 0.7, 528: 0.8})

    def test_off_grid_rows_are_caught_and_neighbours_are_not(self):
        step = (self.chips["maxx"] - self.chips["minx"]).median() / 2
        import geopandas as gpd
        neighbour = self.squares.geometry.iloc[[0]].translate(yoff=step)    # next tile's chip
        stray = self.squares.geometry.iloc[[100]].translate(xoff=step / 3)  # off the grid
        rows = gpd.GeoDataFrame(geometry=pd.concat([neighbour, stray]), crs="EPSG:4326")
        amw = self.amw_rows(rows, [0.9, 0.9])
        _, unmatched = compare.match(self.chips, amw)
        flagged = compare.should_have_matched(unmatched, self.chips)
        self.assertEqual(flagged.tolist(), [False, True])

    def test_tile_number_is_chip_id_without_the_chip(self):
        number = int(self.chips["chip_id"].iloc[0]) // 1000
        self.assertEqual(compare.tile_number(self.tile), number)
        self.assertEqual(compare.key_number(TILE), number)


class TestOutcome(unittest.TestCase):
    def table(self):
        return pd.DataFrame({
            "confidence":     [0.9, 0.9, 0.1, 0.1, 0.4, 0.41],
            "amw_confidence": [0.8, np.nan, 0.7, np.nan, 0.45, 0.4],
        })

    def test_labels(self):
        got = compare.outcome(self.table()).tolist()
        # 0.4 itself is not a detection: AMW's cut is strictly greater (inference_engine.py:803)
        self.assertEqual(got, ["both", "only ours", "only AMW", "neither", "only AMW", "only ours"])

    def test_agreement(self):
        a = compare.agreement(self.table())
        self.assertEqual(a["both"], 1)
        self.assertEqual(a["ours total"], 3)
        self.assertEqual(a["AMW total"], 3)
        self.assertAlmostEqual(a["overlap (Jaccard)"], 1 / 5)

    def test_score_gap(self):
        t = pd.DataFrame({"confidence": [0.5, 0.6, 0.7, 0.1],
                          "amw_confidence": [0.5, 0.605, 0.75, np.nan]})
        g = compare.score_gap(t)
        self.assertEqual(g["chips"], 3)
        self.assertAlmostEqual(g["identical (gap <= 1e-6)"], 1 / 3)
        self.assertAlmostEqual(g["gap <= 0.01"], 2 / 3)
        self.assertAlmostEqual(g["largest gap"], 0.05)


class TestRefusals(unittest.TestCase):
    def test_duplicate_amw_location(self):
        chips = pd.DataFrame({"amw_key_x": [1.0, 2.0], "amw_key_y": [1.0, 1.0]})
        amw = pd.DataFrame({"amw_key_x": [1.0, 1.0], "amw_key_y": [1.0, 1.0],
                            "amw_confidence": [0.5, 0.6]})
        with self.assertRaises(ValueError):
            compare.match(chips, amw)


class TestVotes(unittest.TestCase):
    def test_parse_votes(self):
        published = "[7.9042510e-03 8.6182344e-01 3.5066527e-01 1.0675349e-25 9.8146520e-01\n 5.9426147e-01]"
        with_commas = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]"
        v = compare.parse_votes(pd.Series([published, with_commas]))
        self.assertEqual(v.shape, (2, 6))
        self.assertAlmostEqual(v[0, 3], 1.0675349e-25)
        self.assertAlmostEqual(v[0, 5], 0.59426147)
        self.assertAlmostEqual(v[1].mean(), 0.35)


class TestReadOurs(unittest.TestCase):
    def write_run(self, root: Path) -> None:
        folder = compare.task_folder(root, "2026Q2", 3)
        folder.mkdir(parents=True)
        for i in (1, 2):
            rows = {"chip_id": [i, i + 10], "confidence": [0.5, 0.1], "n_obs": [9.0, 9.0],
                    "n_clear": [4.0, 4.0], "empty_share": [0.0, 0.0]}
            rows |= {f"vote_{k}": [0.5, 0.1] for k in range(1, 7)}
            pd.DataFrame(rows).to_parquet(folder / f"part_{i:05d}.parquet")
        (folder / "_status.json").write_text('{"complete": true}')

    def test_reads_all_parts(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_run(Path(d))
            ours = compare.read_ours(Path(d), "2026Q2", 3)
            self.assertEqual(sorted(ours["chip_id"]), [1, 2, 11, 12])
            self.assertNotIn("vote_1", ours)
            self.assertTrue(compare.read_status(Path(d), "2026Q2", 3)["complete"])

    def test_only_some_chips_with_votes(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_run(Path(d))
            ours = compare.read_ours(Path(d), "2026Q2", 3, votes=True, only=[2, 11])
            self.assertEqual(sorted(ours["chip_id"]), [2, 11])
            self.assertIn("vote_6", ours)

    def test_missing_run_is_clear(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            compare.read_ours(Path(d), "2026Q2", 3)


class TestSceneDates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import scene_dates
        cls.sd = scene_dates

    def test_windows(self):
        # AMW's quarter ends a day before ours; Earth Engine excludes both ends
        self.assertEqual(self.sd.windows("2026Q3"),
                         (date(2026, 7, 1), date(2026, 9, 30), date(2026, 10, 1)))
        self.assertEqual(self.sd.windows("2026Q4")[2], date(2027, 1, 1))

    def test_report_names_the_last_day(self):
        utc = self.sd.utc
        table = pd.DataFrame({
            "scene": ["A", "B", "C"],
            "taken": utc([1789000000000, 1790700000000, 1790760000000]),   # Sep 9, 29, 30
            "cloud_pct": [10.0, 50.0, 5.0],
            "ingested": utc([1789100000000000, 1790800000000000, None], "us"),
            "cloud_score_ingested": utc([1789100000000000, None, 1790900000000000], "us"),
        })
        text = self.sd.report(table, "2026Q3")
        self.assertIn("AMW   2026-07-01 to 2026-09-30 (excluded): 2 scenes", text)
        self.assertIn("last day 2026-09-30: 1 scene(s)", text)
        self.assertIn("C  cloud 5%  Cloud Score+ yes", text)
        self.assertIn("scenes without a Cloud Score+ image: 1", text)


if __name__ == "__main__":
    unittest.main()
