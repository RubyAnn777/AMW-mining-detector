"""The analysis tables build from a made-up panel, and notebooks 1 to 4 run on them without a failing cell.

    python -m unittest panel_pipeline/checks/tests/test_analysis.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_tables as at  # noqa: E402
from fake_panel import MONTHS, QUARTERS, make_panel_run  # noqa: E402

NOTEBOOKS = ["1_clouds", "2_persistence", "3_distance", "4_frequencies"]


class HelpersTest(unittest.TestCase):
    def test_patterns_gaps_and_retention(self):
        m = pd.DataFrame([[1, 0, 0, 1], [1, np.nan, 1, 0], [0, 0, 0, 0]], index=[10, 11, 12],
                         columns=["2025Q1", "2025Q2", "2025Q3", "2025Q4"], dtype=float)
        self.assertEqual(list(at.pattern_strings(m)), ["DMMD", "D·DM", "MMMM"])
        gaps = at.gap_table(m)
        closed = gaps[gaps["closed"]]
        self.assertEqual(list(closed["chip_id"]), [10, 10])          # chip 10: D M M D, one row per missed period
        self.assertEqual(list(closed["length"]), [2, 2])
        self.assertEqual(list(gaps.loc[~gaps["closed"], "chip_id"]), [11])   # chip 11 ends missed; 12 never detected
        r = at.retention(m).set_index(["from", "to"])
        self.assertEqual(int(r.loc[("2025Q1", "2025Q2"), "detected in from"]), 1)   # chip 11 is not scored in Q2: left out
        self.assertEqual(r.loc[("2025Q1", "2025Q3"), "share"], 0.5)

    def test_wilson_and_auc(self):
        lo, hi = at.wilson(np.array([5]), np.array([10]))
        self.assertLess(lo[0], 0.5)
        self.assertGreater(hi[0], 0.5)
        self.assertAlmostEqual(at.auc(np.array([0, 1]), np.array([2, 3])), 1.0)
        self.assertAlmostEqual(at.auc(np.array([1, 2]), np.array([1, 2])), 0.5)

    def test_amw_yearly_rule(self):
        amw = pd.DataFrame({"chip_id": [1, 1, 2, 3, 4], "year": [2024, 2025, 2025, 2025, 2023],
                            "frequency": "yearly", "det_loose": True, "det_stringent": [True, True, True, False, True]})
        s = at.yearly_status(amw).set_index("chip_id")["status"]
        self.assertEqual(s[1], "confirmed")
        self.assertEqual(s[2], "provisional (2025 only)")
        self.assertEqual(s[3], "2025 only, below the provisional rule")
        self.assertEqual(s[4], "never confirmed")

    def test_period_spans(self):
        self.assertEqual(at.period_span("2024"), (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")))
        self.assertEqual(at.period_span("2025Q3")[1], pd.Timestamp("2025-10-01"))
        self.assertEqual(at.months_of("2026Q1"), ["2026-01", "2026-02", "2026-03"])


class BuildAndNotebooksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = make_panel_run(Path(cls.tmp.name) / "fake_panel")
        cls.build = subprocess.run([sys.executable, str(REPO / "notebooks" / "build_tables.py"), "--run", str(cls.run_dir),
                                    "--amw-dir", str(cls.run_dir / "amw")], capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tables_are_complete(self):
        self.assertEqual(self.build.returncode, 0, self.build.stdout + self.build.stderr)
        a = at.Analysis(self.run_dir)
        self.assertTrue(a.ready)
        self.assertEqual(a.periods("quarterly"), QUARTERS)
        self.assertEqual(a.periods("monthly"), MONTHS)
        # every chip detected anywhere is followed, so counts from the hits are complete
        for tag in QUARTERS:
            post = pd.read_parquet(self.run_dir / "panel" / "quarterly" / "postprocess" / f"period={tag}" / "data.parquet")
            hits = a.table("quarterly", "hits", [tag])
            self.assertEqual(int(post["det_raw"].fillna(False).sum()), int(hits["det_raw"].fillna(False).sum()), tag)
        tiles = a.table("quarterly", "tiles", QUARTERS[:1])
        self.assertEqual(int(tiles["chips"].sum()), len(pd.read_parquet(self.run_dir / "plan" / "chips.parquet")))

    def test_a_second_build_reads_nothing_new(self):
        before = json.loads((self.run_dir / "analysis" / "_build.json").read_text())
        again = subprocess.run([sys.executable, str(REPO / "notebooks" / "build_tables.py"), "--run", str(self.run_dir),
                                "--amw-dir", str(self.run_dir / "amw")], capture_output=True, text=True)
        self.assertEqual(again.returncode, 0, again.stderr)
        after = json.loads((self.run_dir / "analysis" / "_build.json").read_text())
        self.assertEqual(before["static_built_utc"], after["static_built_utc"])
        self.assertEqual(before["periods"], after["periods"])          # no period summarised again

    def test_every_notebook_runs_without_a_failing_cell(self):
        self.assertEqual(self.build.returncode, 0, self.build.stderr)
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                out = Path(self.tmp.name) / f"{name}.html"
                result = subprocess.run([sys.executable, "-m", "marimo", "export", "html", "--no-include-code",
                                         str(REPO / "notebooks" / f"{name}.py"), "-o", str(out), "--", "--run", str(self.run_dir)],
                                        capture_output=True, text=True, cwd=self.tmp.name)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("failed to execute", result.stdout + result.stderr)
                self.assertLess(out.stat().st_size, 30e6)


if __name__ == "__main__":
    unittest.main()
