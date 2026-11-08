"""Tests for last_day_check.py on a small made-up run. No Earth Engine, no TensorFlow:
the last day's scenes and the re-scoring are stand-ins.

    python -m unittest discover -s amw_comparison/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                   # compare.py, last_day*.py
sys.path.insert(1, str(HERE.parents[1] / "panel_pipeline" / "checks" / "tests"))   # fake_run.py
for _name in ("ee", "gee"):                    # last_day.py imports Earth Engine; not used here
    sys.modules.setdefault(_name, types.ModuleType(_name))

import compare  # noqa: E402
import last_day_check as ldc  # noqa: E402
from fake_run import make_run  # noqa: E402

QUARTER = "2025Q1"


class FakeRescorer:
    """Scores as if the last day were the only difference: AMW's dates give AMW's file
    (and below 0.4 where AMW has no row), our dates give our stored scores."""

    def __init__(self, run, amw_scores):
        self.run, self.amw = run, amw_scores

    def score(self, tile_key, kind, start, end):
        tile = compare.key_number(tile_key)
        sub = int(pd.read_csv(self.run / "plan" / "tiles.csv").set_index("tile_key").loc[[tile_key], "subregion"].min())
        ours = ldc.our_scores(self.run, QUARTER, sub, tile).sort_values("chip_id")
        if end.endswith("-31"):                         # AMW's end date for Q1: 31 March
            amw = ours["chip_id"].map(self.amw)
            return np.where(amw.notna(), amw, np.minimum(ours["ours_stored"], 0.39)).astype(float)
        return ours["ours_stored"].to_numpy()


class TestLastDayCheck(unittest.TestCase):

    def test_quarter(self):
        with tempfile.TemporaryDirectory() as d:
            run = make_run(Path(d) / "run", panel=False, comparison=False)
            amw_dir = Path(d) / "amw"
            amw_dir.mkdir()
            # AMW's file: our detections, except that the western half (under the "last
            # day's scenes") has slightly different scores
            plan = pd.read_parquet(run / "plan" / "chips.parquet")
            parts = pd.concat(pd.read_parquet(p, columns=["chip_id", "confidence"]) for p in
                              (run / "detect" / "quarterly" / f"period={QUARTER}").glob("subregion=*/part_*.parquet"))
            t = plan.merge(parts.drop_duplicates("chip_id"), on="chip_id")
            west = t["minx"] < t["minx"].median()
            t["amw"] = np.where(west, t["confidence"] - 0.03, t["confidence"])
            t = t[t["amw"] > 0.4]
            gpd.GeoDataFrame({"confidence": t["amw"].to_numpy()},
                             geometry=shapely.box(*(t[c].to_numpy() for c in ("minx", "miny", "maxx", "maxy"))),
                             crs="EPSG:4326").to_file(amw_dir / f"amazon_basin_Q125_t0.4.geojson",
                                                      driver="GeoJSON", COORDINATE_PRECISION=6)
            compare.compare_run(run, [QUARTER], amw_dir=amw_dir)
            chips = pd.read_parquet(run / "amw_comparison" / f"chips_{QUARTER}.parquet")
            cut = plan["minx"].median()
            outline = shapely.box(-180, -90, cut, 90)          # the "last day's scenes"

            summary = ldc.check_quarter(
                run, QUARTER, tiles_per_subregion=2, subregions=None,
                scenes_fn=lambda c, q: (pd.DataFrame({"scene": ["x"], "cloud_pct": [5]}), outline),
                rescorer=FakeRescorer(run, chips.set_index("chip_id")["amw_confidence"]))
            out = run / "amw_comparison" / "last_day_check" / QUARTER
            table = pd.read_csv(out / "by_subregion.csv")
            rescored = pd.read_csv(out / "rescore.csv")
            outside = pd.read_csv(out / "outside.csv")

        text = "\n".join(summary)
        self.assertIn("1. Share", text)
        self.assertEqual(set(table["outcome"]), set(ldc.OUTCOMES))
        differ = table[table["outcome"] == "both, scores differ"]
        self.assertEqual(int(differ["elsewhere"].sum()), 0)       # every gap lies in the west
        self.assertTrue(len(rescored) > 0)
        self.assertTrue(rescored[["a ok", "b ok", "c ok"]].all().all())
        self.assertNotIn("NOT ALL", text)
        self.assertTrue((outside["square_touches"] | (outside["outcome"] != "both, scores differ")).all()
                        or outside.empty)

    def test_outside_tiles_are_picked_by_their_gaps(self):
        # tile 1: disagreement under the last day; tile 2: agreement elsewhere (control);
        # tiles 3 and 4: disagreements far from any last-day scene, tile 4 with more of them
        rows = [(1, "both, scores differ", True, True, 0.05)] * 3 \
             + [(2, "both, same score", False, False, 0.0)] * 3 \
             + [(3, "both, scores differ", False, False, 0.02)] * 2 \
             + [(4, "both, scores differ", False, False, 0.03)] * 5 \
             + [(4, "only ours", False, False, np.nan)] \
             + [(5, "both, scores differ", False, False, 0.0005)] * 9      # tiny gaps: not picked
        t = pd.DataFrame(rows, columns=["tile", "outcome", "centre_under", "square_touches", "gap"])
        t["subregion"] = 1
        picks = ldc.pick_tiles(t, per_subregion=1, subregions=None, outside_per_subregion=1)
        self.assertEqual(dict(zip(picks["why"], picks["tile"])),
                         {"disagreement under the last day": 1, ldc.CONTROL: 2, ldc.OUTSIDE: 4})
        none = ldc.pick_tiles(t, per_subregion=1, subregions=None)
        self.assertNotIn(ldc.OUTSIDE, set(none["why"]))


if __name__ == "__main__":
    unittest.main()
