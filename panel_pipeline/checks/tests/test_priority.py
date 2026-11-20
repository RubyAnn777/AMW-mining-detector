"""Tests for two-stage scoring: priority.py, lib/priority.py and status.py --priority.
On a small made-up run (fake_run.py). No Earth Engine, no TensorFlow.

    python -m unittest discover -s panel_pipeline/checks/tests -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPELINE / "lib"))
import priority as prio  # noqa: E402
from fake_run import make_run, tiles  # noqa: E402

RING_OF_COLUMN = {0: 0, 1: 1, 2: 2, 3: 4}     # tile column ti -> its ring around AMW's tiles


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PIPELINE / script), *args],
                          capture_output=True, text=True)


def fake_history(root: Path) -> None:
    """What amw_comparison/history.py writes, for the fake run's 12 tiles."""
    t = tiles().drop_duplicates("tile_key")
    out = root / "amw_comparison"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"tile": t["tile_key"].map(prio.number_of_key), "tile_key": t["tile_key"],
                  "ring": t["ti"].map(RING_OF_COLUMN).astype("int8")}).to_parquet(out / "history_tiles.parquet")
    (out / "history_summary.json").write_text(json.dumps({
        "files": {"2024": "amazon_basin_2024_t0.4.geojson"}, "fingerprints": {"2024": "abc"},
        "amw_locations": 10, "matched_to_our_chips": 10, "max_ring": 3}))


class TestPriority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_run(Path(self.tmp.name) / "fake_run", panel=False, comparison=False)
        fake_history(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, rings="1"):
        r = run("priority.py", "--run", str(self.root), "--rings", rings)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r

    def test_list_is_written_once_and_checked(self):
        self.assertIsNone(prio.load(self.root))
        self.assertIn("nothing written", run("priority.py", "--run", str(self.root), "--rings", "1",
                                             "--dry-run").stdout)
        self.assertFalse(prio.exists(self.root))
        self.make("1")
        p = prio.load(self.root)
        t = tiles().drop_duplicates("tile_key")
        expected = {prio.number_of_key(k): 1 if ti <= 1 else 2 for k, ti in zip(t["tile_key"], t["ti"])}
        self.assertEqual(p.of, expected)
        self.assertEqual(p.rings, 1)
        # the same list again: fine, nothing changes; another one: refused
        self.assertIn("exactly this list", self.make("1").stdout)
        other = run("priority.py", "--run", str(self.root), "--rings", "2")
        self.assertNotEqual(other.returncode, 0)
        self.assertIn("move", other.stderr + other.stdout)
        # an edited list is refused everywhere
        csv = prio.folder(self.root) / prio.TILES
        csv.write_text(csv.read_text().replace(",1\n", ",2\n", 1))
        with self.assertRaises(prio.PriorityError):
            prio.load(self.root)
        self.assertNotEqual(run("status.py", "--run", str(self.root)).returncode, 0)

    def test_a_used_list_cannot_be_replaced(self):
        self.make("1")
        status = self.root / "detect" / "monthly" / "period=2025-01" / "subregion=2" / "_status.json"
        s = json.loads(status.read_text())
        s["priority"] = {"list": prio.load(self.root).tag}
        status.write_text(json.dumps(s))
        r = run("priority.py", "--run", str(self.root), "--rings", "2")
        self.assertIn("can no longer change", r.stderr + r.stdout)

    def test_counts(self):
        self.make("1")
        p = prio.load(self.root)
        numbers = [prio.number_of_key(k) for k in tiles().query("subregion == 2")["tile_key"]]
        c = p.counts(numbers, set(numbers[:2]), {numbers[-1]})
        self.assertEqual(c["1"], {"planned": 6, "saved": 2, "failed": 0})
        self.assertEqual(c["2"], {"planned": 3, "saved": 0, "failed": 1})

    def test_status_todo_per_priority(self):
        self.make("1")
        tag = prio.load(self.root).tag
        # 2025-05 subregion 3: priority 1 done by a --priority 1 session (new-style counts)
        folder = self.root / "detect" / "monthly" / "period=2025-05" / "subregion=3"
        folder.mkdir(parents=True)
        (folder / "_status.json").write_text(json.dumps({
            "complete": False, "tiles_planned": 6, "tiles_saved": 3, "chips_saved": 3 * 529,
            "tiles_not_tried": 3, "tiles_failed": {}, "sessions": [{"limit": None}],
            "priority": {"list": tag, "rings": 1, "1": {"planned": 3, "saved": 3, "failed": 0},
                         "2": {"planned": 3, "saved": 0, "failed": 0}}}))
        p1 = run("status.py", "--run", str(self.root), "--priority", "1", "--todo").stdout.splitlines()
        plain = run("status.py", "--run", str(self.root), "--todo").stdout.splitlines()
        self.assertTrue(all(line.endswith("--priority 1") for line in p1))
        self.assertFalse(any("--from 2025-05 --to 2025-05" in l and "subregion 3" in l for l in p1))
        self.assertTrue(any("--from 2025-05 --to 2025-05" in l and "subregion 3" in l for l in plain))
        # 2025-04 subregion 2 (2 of its 6 priority-1 tiles saved, old-style status): still to do
        self.assertTrue(any("--subregion 2 --from 2025-04" in l for l in p1))
        self.assertFalse(any("2025-01" in l for l in p1))          # complete months are done
        p2 = run("status.py", "--run", str(self.root), "--priority", "2", "--todo").stdout
        self.assertIn("--subregion 3 --from 2025-05 --to 2025-05 --priority 2", p2)
        overview = run("status.py", "--run", str(self.root))
        self.assertEqual(overview.returncode, 0, overview.stderr)
        self.assertIn("priority-1 tiles saved", overview.stdout)
        shown = run("status.py", "--run", str(self.root), "--priority", "1", "--all")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("priority 1 only", shown.stdout)

    def test_counts_from_part_files(self):
        self.make("1")
        sys.path.insert(0, str(PIPELINE))
        import status as st
        p = prio.load(self.root)
        numbers = st.tile_numbers(self.root)
        c = st.priority_counts(self.root / "detect" / "monthly" / "period=2025-04" / "subregion=2", p, numbers[2])
        self.assertEqual(c["1"]["saved"], 2)
        self.assertEqual(c["2"]["saved"], 0)


if __name__ == "__main__":
    unittest.main()
