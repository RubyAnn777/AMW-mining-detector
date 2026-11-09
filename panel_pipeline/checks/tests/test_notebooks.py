"""The marimo notebooks run from top to bottom on a made-up run, and their summaries cache.

    python -m unittest discover -s panel_pipeline/checks/tests
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from notebook_kit import cached, find_runs, is_cached, thin  # noqa: E402
from fake_run import make_run  # noqa: E402

NOTEBOOKS = ["panel_overview", "explore_runs", "amw_comparison", "run_performance"]


class CachedTest(unittest.TestCase):
    def test_computed_once_then_reused_until_an_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, calls = Path(tmp), []
            source = run / "input.txt"
            source.write_text("a")
            compute = lambda: calls.append(1) or pd.DataFrame({"x": [len(calls)]})
            self.assertEqual(cached(run, "t", compute, [source])["x"][0], 1)
            self.assertTrue(is_cached(run, "t", [source]))
            self.assertEqual(cached(run, "t", compute, [source])["x"][0], 1)    # stored, not recomputed
            later = time.time() + 5
            os.utime(source, (later, later))                                    # the input changed
            self.assertFalse(is_cached(run, "t", [source]))
            self.assertEqual(cached(run, "t", compute, [source])["x"][0], 2)
            self.assertEqual(len(calls), 2)


class ThinTest(unittest.TestCase):
    def test_at_most_n_rows_and_a_line_per_task_stays_in_order(self):
        d = pd.DataFrame({"task": np.repeat(["a", "b"], 30_000), "t": np.tile(np.arange(30_000), 2)})
        sample, note = thin(d, 10_000)
        self.assertEqual(len(sample), 10_000)
        self.assertIn("of 60,000", note)
        lines, _ = thin(d, 10_000, every="task")
        self.assertLessEqual(len(lines), 10_000)
        self.assertTrue((lines.groupby("task")["t"].diff().dropna() > 0).all())
        self.assertEqual(thin(d.head(5))[1], "")


class NotebooksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = make_run(Path(cls.tmp.name) / "fake_run", panel=False)
        for frequency in ("monthly", "quarterly"):
            subprocess.run([sys.executable, str(REPO / "panel_pipeline" / "assemble_period.py"), "--run", str(cls.run_dir),
                            "--frequency", frequency], capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_picker_finds_the_run(self):
        self.assertEqual(list(find_runs(self.run_dir)), ["fake_run"])

    def test_every_notebook_runs_without_a_failing_cell(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                out = Path(self.tmp.name) / f"{name}.html"
                result = subprocess.run([sys.executable, "-m", "marimo", "export", "html", str(REPO / "notebooks" / f"{name}.py"),
                                         "-o", str(out), "--", "--run", str(self.run_dir)],
                                        capture_output=True, text=True, cwd=self.tmp.name)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("failed to execute", result.stdout + result.stderr)
        stored = {p.stem for p in (self.run_dir / "summaries").glob("*.pkl")}
        self.assertTrue({"checks_panel", "seen_panel", "grid", "tasks"} <= stored, stored)

    def test_a_full_size_comparison_fits_in_the_browser(self):
        # 9 Nov 2026: the AMW score chart sent every chip of a real quarter, 260 MB, and
        # marimo refused to show it. Blow the made-up comparison up to that size.
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp) / "big_run", panel=False)
            rng = np.random.default_rng(0)
            for path in (run / "amw_comparison").glob("chips_*.parquet"):
                d = pd.read_parquet(path).sample(400_000, replace=True, random_state=0).reset_index(drop=True)
                d["chip_id"] = np.arange(len(d)) + 10**9
                d["confidence"] = rng.uniform(0.3, 1, len(d))
                d["amw_confidence"] = np.where(rng.random(len(d)) < 0.7, d["confidence"], rng.uniform(0.3, 1, len(d)))
                d["amw_key_x"], d["amw_key_y"] = rng.uniform(-70, -50, len(d)), rng.uniform(-15, 5, len(d))
                d.to_parquet(path)
            out = Path(tmp) / "amw.html"
            result = subprocess.run([sys.executable, "-m", "marimo", "export", "html",
                                     str(REPO / "notebooks" / "amw_comparison.py"), "-o", str(out), "--", "--run", str(run)],
                                    capture_output=True, text=True, cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("too large", out.read_text())


if __name__ == "__main__":
    unittest.main()
