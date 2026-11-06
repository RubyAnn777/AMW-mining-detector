"""Tests for compare.py and last_day.py. No Earth Engine, no downloads.

    python -m unittest discover -s amw_comparison/tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                            # compare.py, last_day.py
sys.path.insert(1, str(HERE.parent.parent / "panel_pipeline" / "lib"))  # grid.py, for real chips

import compare  # noqa: E402

TILE = "552:12:10.0:21:60:-100"   # a subregion 3 tile; DLTile.from_key needs no network


def real_chips():
    """One real tile's chips, as plan.py writes them, and their squares."""
    from descarteslabs.geo import DLTile
    from grid import chip_table
    squares = chip_table(DLTile.from_key(TILE), stride=24)
    chips = pd.DataFrame(squares.drop(columns="geometry")).assign(tile=lambda c: c["chip_id"] // 1000)
    return chips, squares


def write_amw(squares, confidence, path: Path) -> Path:
    """As AMW writes its files: squares in EPSG:4326, 6 decimals (concatenate.py)."""
    squares[["geometry"]].assign(confidence=confidence).to_file(
        path, driver="GeoJSON", COORDINATE_PRECISION=6)
    return path


class TestJoinOnRealChips(unittest.TestCase):
    """The join must find every chip again after a round trip through an AMW-style file."""

    @classmethod
    def setUpClass(cls):
        cls.chips, cls.squares = real_chips()

    def amw_rows(self, squares, confidence) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as d:
            return compare.read_amw(write_amw(squares, confidence, Path(d) / "amw.geojson"))

    def test_file_name(self):
        self.assertEqual(compare.amw_file("2026Q3").name, "amazon_basin_Q326_t0.4.geojson")

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
        import geopandas as gpd
        step = (self.chips["maxx"] - self.chips["minx"]).median() / 2
        neighbour = self.squares.geometry.iloc[[0]].translate(yoff=step)    # next tile's chip
        stray = self.squares.geometry.iloc[[100]].translate(xoff=step / 3)  # off the grid
        rows = gpd.GeoDataFrame(geometry=pd.concat([neighbour, stray]), crs="EPSG:4326")
        _, unmatched = compare.match(self.chips, self.amw_rows(rows, [0.9, 0.9]))
        self.assertEqual(compare.should_have_matched(unmatched, self.chips).tolist(), [False, True])

    def test_tile_number_is_chip_id_without_the_chip(self):
        self.assertEqual(compare.key_number(TILE), int(self.chips["chip_id"].iloc[0]) // 1000)

    def test_duplicate_amw_location_is_refused(self):
        amw = pd.DataFrame({"amw_key_x": [1.0, 1.0], "amw_key_y": [1.0, 1.0], "amw_confidence": [0.5, 0.6]})
        with self.assertRaises(ValueError):
            compare.match(self.chips, amw)


class TestCounting(unittest.TestCase):
    def table(self):
        return pd.DataFrame({
            "confidence":     [0.9, 0.9, 0.1, 0.1, 0.4, 0.41],
            "amw_confidence": [0.8, np.nan, 0.7, np.nan, 0.45, 0.4],
        })

    def test_labels(self):
        # 0.4 itself is not a detection: AMW's cut is strictly greater (inference_engine.py:803)
        self.assertEqual(compare.outcome(self.table()).tolist(),
                         ["both", "only ours", "only AMW", "neither", "only AMW", "only ours"])

    def test_agreement(self):
        a = compare.agreement(self.table())
        self.assertEqual((a["both"], a["ours total"], a["AMW total"]), (1, 3, 3))
        self.assertAlmostEqual(a["overlap (Jaccard)"], 1 / 5)

    def test_score_gap(self):
        g = compare.score_gap(pd.DataFrame({"confidence": [0.5, 0.6, 0.7, 0.1],
                                            "amw_confidence": [0.5, 0.605, 0.75, np.nan]}))
        self.assertEqual(g["chips"], 3)
        self.assertAlmostEqual(g["identical (gap <= 1e-6)"], 1 / 3)
        self.assertAlmostEqual(g["gap <= 0.01"], 2 / 3)
        self.assertAlmostEqual(g["largest gap"], 0.05)

    def test_one_per_location_keeps_the_higher_score(self):
        rows = pd.DataFrame({"chip_id": [1, 2, 3], "amw_key_x": [1.0, 1.0, 2.0],
                             "amw_key_y": [1.0, 1.0, 1.0], "confidence": [0.5, 0.7, 0.9]})
        got = compare.one_per_location(rows)
        self.assertEqual(got["chip_id"].tolist(), [2, 3])
        self.assertEqual(got["border_copies"].tolist(), [2, 1])
        self.assertAlmostEqual(got["border_spread"].iloc[0], 0.2)


class TestWholeRun(unittest.TestCase):
    """compare_run on a one-tile run: two subregions both list the tile (a border tile),
    the quarter and its three months are scored, AMW detects a few chips."""

    def test_compare_run(self):
        chips, squares = real_chips()
        with tempfile.TemporaryDirectory() as d:
            run, amw_dir = Path(d) / "run", Path(d) / "amw"
            (run / "plan").mkdir(parents=True)
            amw_dir.mkdir()
            chips.to_parquet(run / "plan" / "chips.parquet")
            pd.DataFrame({"subregion": [2, 3], "tile_key": [TILE, TILE]}).to_csv(run / "plan" / "tiles.csv")
            conf = np.full(len(chips), 0.1)
            conf[:3] = [0.9, 0.8, 0.45]                      # we detect chips 0, 1, 2
            for subregion, shift in ((2, 0.0), (3, 0.01)):   # subregion 3 scores chip 0 higher
                self.write_task(run, "quarterly", "2025Q1", subregion, chips, conf + shift * (np.arange(len(conf)) == 0))
                for month in ("2025-01", "2025-02", "2025-03"):
                    self.write_task(run, "monthly", month, subregion, chips, conf * (month != "2025-02"))
            write_amw(squares.iloc[[0, 1, 5]], [0.9, 0.8, 0.6], amw_dir / "amazon_basin_Q125_t0.4.geojson")

            out = compare.compare_run(run, ["2025Q1"], amw_dir=amw_dir)
            got = pd.read_parquet(out / "chips_2025Q1.parquet")
            got.index = got["chip_id"] % 1000                  # the chip within its tile
            joins = pd.read_csv(out / "joins.csv")
            inputs = pd.read_csv(out / "inputs.csv")

        self.assertEqual(sorted(got.index), [0, 1, 2, 5])       # detected by someone
        self.assertEqual(compare.outcome(got).to_dict(), {0: "both", 1: "both", 2: "only ours", 5: "only AMW"})
        self.assertAlmostEqual(got.loc[0, "confidence"], 0.91)  # the higher of the two copies
        self.assertEqual(got.loc[0, "subregion"], 3)
        self.assertEqual(got.loc[0, "border_copies"], 2)
        self.assertAlmostEqual(got.loc[0, "border_spread"], 0.01)
        self.assertEqual(got.loc[0, "months_detected"], 2)      # February scored 0
        self.assertEqual(got.loc[0, "months_scored"], 3)
        self.assertEqual(joins["should have matched"].tolist(), [0, 0])
        self.assertEqual(len(inputs), 8)
        self.assertTrue((inputs["state"] == "complete").all())

    @staticmethod
    def write_task(run, frequency, period, subregion, chips, conf):
        folder = compare.task_folder(run, period, subregion, frequency)
        folder.mkdir(parents=True)
        half = len(chips) // 2                                  # two part files, as while unmerged
        scores = pd.DataFrame({"chip_id": chips["chip_id"], "confidence": conf, "n_obs": 9.0,
                               "n_clear": 4.0, "empty_share": 0.0})
        scores.iloc[:half].to_parquet(folder / "part_00001.parquet")
        scores.iloc[half:].to_parquet(folder / "part_00002.parquet")
        (folder / "_status.json").write_text(json.dumps(
            {"complete": True, "tiles_saved": 1, "tiles_planned": 1, "tiles_failed": {}}))


class TestYears(unittest.TestCase):
    """A year's quarters, and results that later years add to instead of replacing."""

    def test_quarters_of_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run").mkdir()
            (run / "run" / "config.yaml").write_text(
                "periods: {first_month: '2018-01', last_month: '2026-09'}\n")
            self.assertEqual(compare.quarters_of_year(run, 2026), ["2026Q1", "2026Q2", "2026Q3"])
            self.assertEqual(compare.quarters_of_year(run, 2025), [f"2025Q{i}" for i in range(1, 5)])
            self.assertEqual(compare.quarters_of_year(run, 2027), [])

    def test_write_merged_keeps_other_years(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joins.csv"
            key = ["quarter", "subregion"]
            compare.write_merged(pd.DataFrame({"quarter": ["2026Q1", "2026Q1"], "subregion": [1, 2],
                                               "n": [1, 2]}), path, key)
            compare.write_merged(pd.DataFrame({"quarter": ["2025Q4"], "subregion": [1], "n": [3]}),
                                 path, key)
            compare.write_merged(pd.DataFrame({"quarter": ["2026Q1"], "subregion": [1], "n": [9]}),
                                 path, key)       # 2026Q1 again: replaces its own row only
            got = pd.read_csv(path)
            self.assertEqual(got.values.tolist(),
                             [["2025Q4", 1, 3], ["2026Q1", 1, 9], ["2026Q1", 2, 2]])


class TestLastDay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import last_day
        cls.ld = last_day

    def test_windows(self):
        # AMW's quarter ends a day before ours; Earth Engine excludes both ends
        self.assertEqual(self.ld.windows("2026Q3"), (date(2026, 7, 1), date(2026, 9, 30), date(2026, 10, 1)))
        self.assertEqual(self.ld.windows("2026Q4")[2], date(2027, 1, 1))

    def test_under_last_day(self):
        import shapely
        chips = pd.DataFrame({
            "amw_key_x":      [0.5, 0.5, 0.5, 5.0, 5.0],
            "amw_key_y":      [0.5, 0.5, 0.5, 5.0, 5.0],
            "confidence":     [0.9, 0.9, 0.9, 0.9, 0.1],
            "amw_confidence": [0.9, 0.8, np.nan, 0.9, 0.7],
        })
        t = self.ld.under_last_day(chips, shapely.box(0, 0, 1, 1))
        self.assertEqual(t.loc["both"].iloc[:2].tolist(), [2, 1])
        self.assertEqual(t.loc["both, scores differ"].iloc[:2].tolist(), [1, 0])
        self.assertEqual(t.loc["only ours"].iloc[:2].tolist(), [1, 0])
        self.assertEqual(t.loc["only AMW"].iloc[:2].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
