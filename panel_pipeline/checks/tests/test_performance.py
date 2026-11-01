"""Tests for checks/performance.py on a small made-up run folder. No Earth Engine, no cluster.

    python -m unittest discover -s panel_pipeline/checks/tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # checks/
import performance as perf                                          # noqa: E402

LINE = ("{t}  {p}  try 1: {saved}/{planned} tiles saved (+{added}, 0 failed) | {rate} tiles/min, "
        "download median {dl} s (slowest {slow} s), workers busy {wb}%, scoring busy {sb}%, "
        "{retries} retries | 3 downloading, 0 waiting for a worker, 2 waiting to be scored, "
        "oldest asked for 9 s ago | about 2.0 h left in this try")


def make_run(root: Path) -> None:
    """Two tasks: a month scored in one go, a quarter stopped and resumed (two sessions)."""
    month = root / "detect" / "monthly" / "period=2025-01" / "subregion=3"
    month.mkdir(parents=True)
    (month / "_log.txt").write_text("\n".join([
        "2026-10-28T12:42:00+00:00  2025-01  starting: 1,000 tiles to score, 3 workers, a part every 500 tiles or 2 min",
        LINE.format(t="2026-10-28T12:44:00+00:00", p="2025-01", saved="60", planned="1,000", added=60,
                    rate=30, dl=5, slow=12, wb=90, sb=95, retries=0),
        LINE.format(t="2026-10-28T12:46:00+00:00", p="2025-01", saved="120", planned="1,000", added=60,
                    rate=30, dl=6, slow=14, wb=96, sb=100, retries=2),
        "2026-10-28T12:46:01+00:00  slow 123:456:789: download 45 s",
        "2026-10-28T13:15:00+00:00  2025-01  try 1: 1,000/1,000 tiles saved (+880, 0 failed) | no tile finished in 2.0 min, 0 retries | 0 downloading, 0 waiting for a worker, 0 waiting to be scored, oldest asked for 0 s ago",
        "2026-10-28T13:15:30+00:00  2025-01  session finished: 1,000/1,000 tiles saved, 0 failed",
    ]) + "\n")
    (month / "_status.json").write_text(json.dumps({
        "complete": True, "tiles_planned": 1000, "tiles_saved": 1000, "chips_saved": 529000,
        "tiles_not_tried": 0, "tiles_failed": {},
        "sessions": [{"started_utc": "2026-10-28T12:42:00+00:00", "finished_utc": "2026-10-28T13:15:30+00:00",
                      "limit": None, "max_workers": 3}]}))

    quarter = root / "detect" / "quarterly" / "period=2025Q1" / "subregion=2"
    quarter.mkdir(parents=True)
    (quarter / "_log.txt").write_text("\n".join([
        "2026-10-27T10:00:00+00:00  2025Q1  starting: 2,000 tiles to score, 20 workers, a part every 500 tiles or 2 min",
        LINE.format(t="2026-10-27T10:10:00+00:00", p="2025Q1", saved="400", planned="2,000", added=400,
                    rate=40, dl=25, slow=60, wb=100, sb=20, retries=5),
        "2026-10-27T10:11:00+00:00  2025Q1  stopped; every scored tile is saved",
        "2026-10-27T12:00:00+00:00  2025Q1  resuming: 400/2,000 tiles already saved in 1 part file(s), not scored again; 1,600 tiles to score, 20 workers, a part every 500 tiles or 2 min",
        LINE.format(t="2026-10-27T12:40:00+00:00", p="2025Q1", saved="2,000", planned="2,000", added=1600,
                    rate=40, dl=8, slow=20, wb=60, sb=100, retries=0),
    ]) + "\n")
    (quarter / "_status.json").write_text(json.dumps({
        "complete": True, "tiles_planned": 2000, "tiles_saved": 2000, "chips_saved": 1058000,
        "tiles_not_tried": 0, "tiles_failed": {},
        "sessions": [{"started_utc": "2026-10-27T10:00:00+00:00", "finished_utc": None,
                      "stopped_utc": "2026-10-27T10:11:00+00:00", "limit": None, "max_workers": 20},
                     {"started_utc": "2026-10-27T12:00:00+00:00", "finished_utc": "2026-10-27T12:40:00+00:00",
                      "limit": None, "max_workers": 20}]}))

    # sacct, both ways it can be saved; times are Berlin local (UTC+1 on these days)
    (root / "jobs.txt").write_text(
        "JobID|JobName|NodeList|Start|End|Elapsed|State\n"
        "100|amw-detect|node22-006|2026-10-27T10:59:00|2026-10-27T11:12:00|00:13:00|COMPLETED\n"
        "101|amw-detect|node25-003|2026-10-27T12:59:00|2026-10-27T13:41:00|00:42:00|COMPLETED\n"
        "102|amw-detect|node20-005|2026-10-28T13:41:00|2026-10-28T14:16:00|1-00:35:00|COMPLETED\n")
    (root / "jobs_table.txt").write_text(
        "JobID           JobName        NodeList    Elapsed      State \n"
        "------------ ---------- --------------- ---------- ---------- \n"
        "100           amw-detect      node22-006   00:13:00  COMPLETED \n"
        "103          interacti+      node17-001   00:45:41  COMPLETED \n")


class PerformanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        make_run(self.run)

    def tearDown(self):
        self.tmp.cleanup()

    def test_progress_lines(self):
        lines = perf.progress(self.run)
        self.assertEqual(len(lines), 5)                       # 3 month lines (one empty), 2 quarter
        first = lines.iloc[0]
        self.assertEqual((first["frequency"], first["period"], first["subregion"]), ("monthly", "2025-01", 3))
        self.assertEqual((first["saved"], first["planned"], first["rate"], first["dl_median_s"]), (60, 1000, 30, 5))
        self.assertEqual((first["workers_busy"], first["scoring_busy"], first["retries"]), (90, 95, 0))
        self.assertEqual(str(first["time"].tz), "UTC")
        empty = lines[lines["rate"] == 0]
        self.assertEqual(len(empty), 1)                       # "no tile finished" keeps its counts
        self.assertEqual(empty["saved"].iloc[0], 1000)
        self.assertTrue(pd.isna(empty["dl_median_s"].iloc[0]))

    def test_scoring_busy_without_space_still_parses(self):
        rows = perf.parse_progress(LINE.format(t="2026-10-28T12:44:00+00:00", p="2025-01", saved="60",
                                               planned="1,000", added=60, rate=30, dl=5, slow=12,
                                               wb=90, sb=95, retries=0).replace("scoring busy", "scoringbusy"))
        self.assertEqual(rows[0]["scoring_busy"], 95)

    def test_sessions_get_their_tiles_and_statistics(self):
        sess = perf.sessions(self.run, perf.progress(self.run))
        self.assertEqual(len(sess), 3)
        month = sess[sess["frequency"] == "monthly"].iloc[0]
        self.assertEqual(month["how"], "finished")
        self.assertEqual(month["tiles"], 1000)                # starting: from 0 to 1,000
        self.assertAlmostEqual(month["hours"], 33.5 / 60, places=3)
        self.assertEqual(month["rate"], 30)                   # median of the busy lines
        self.assertEqual(month["retries"], 2)
        q1, q2 = [r for _, r in sess[sess["frequency"] == "quarterly"].iterrows()]
        self.assertEqual((q1["how"], q1["tiles"]), ("stopped", 400))
        self.assertEqual((q2["how"], q2["tiles"]), ("finished", 1600))   # resuming: from 400
        self.assertEqual(q2["saved_before"], 400)

    def test_jobs_both_formats_and_assignment(self):
        piped = perf.jobs(self.run / "jobs.txt")
        self.assertEqual(list(piped["job"]), ["100", "101", "102"])
        self.assertEqual(str(piped["start"].iloc[0]), "2026-10-27 09:59:00+00:00")   # Berlin -> UTC
        self.assertAlmostEqual(piped["hours"].iloc[2], 24 + 35 / 60, places=6)
        table = perf.jobs(self.run / "jobs_table.txt")
        self.assertEqual(list(table["node"]), ["node22-006", "node17-001"])
        self.assertNotIn("start", table)                      # no Start column in this one
        sess = perf.assign_jobs(perf.sessions(self.run, perf.progress(self.run)), piped)
        self.assertEqual(list(sess["job"]), ["102", "100", "101"])   # month, then the two quarter sessions
        self.assertEqual(sess["node"].iloc[1], "node22-006")
        unassigned = perf.assign_jobs(sess.drop(columns=["job", "node"]), table)
        self.assertTrue(unassigned["job"].isna().all())

    def test_task_and_job_summaries_and_totals(self):
        lines = perf.progress(self.run)
        sess = perf.assign_jobs(perf.sessions(self.run, lines), perf.jobs(self.run / "jobs.txt"))
        tasks = perf.task_summary(sess)
        self.assertEqual(len(tasks), 2)
        quarter = tasks[tasks["frequency"] == "quarterly"].iloc[0]
        self.assertEqual((quarter["sessions"], quarter["tiles_scored"], quarter["tiles_saved"]), (2, 2000, 2000))
        self.assertAlmostEqual(quarter["hours"], 51 / 60, places=6)
        self.assertAlmostEqual(quarter["tiles_per_min"], 2000 / 51, places=6)
        self.assertEqual(quarter["nodes"], "node22-006, node25-003")
        month = tasks[tasks["frequency"] == "monthly"].iloc[0]
        self.assertEqual(month["limited_by"], "both (balanced)")
        self.assertEqual(perf.limited_by(100, 20), "downloads (Earth Engine or the link)")
        self.assertEqual(perf.limited_by(60, 100), "scoring (the node)")
        self.assertEqual(perf.limited_by(float("nan"), 50), "?")
        jobs = perf.job_summary(sess, perf.jobs(self.run / "jobs.txt"), cores_per_node=40)
        self.assertEqual(list(jobs["tiles"]), [400, 1600, 1000])
        self.assertAlmostEqual(jobs["core_hours"].iloc[0], 13 / 60 * 40, places=6)
        t = perf.totals(tasks, jobs, cores_per_node=40)
        self.assertEqual((t["tasks"], t["tile_periods_saved"], t["jobs"]), (2, 3000, 3))
        self.assertAlmostEqual(t["core_hours"], (13 + 42 + 24 * 60 + 35) / 60 * 40, places=6)
        self.assertAlmostEqual(t["tile_periods_per_core_hour"], 3000 / t["core_hours"], places=6)

    def test_summary_without_logs_or_jobs(self):
        for log in self.run.rglob("_log.txt"):
            log.unlink()
        sess = perf.sessions(self.run, perf.progress(self.run))
        tasks = perf.task_summary(sess)                       # no assign_jobs: still works
        self.assertEqual(len(tasks), 2)
        self.assertTrue(tasks["tiles_scored"].eq(0).all() or tasks["tiles_scored"].isna().all())
        self.assertEqual(list(tasks["limited_by"]), ["?", "?"])
        self.assertEqual(perf.totals(tasks, None)["tile_periods_saved"], 3000)
        self.assertTrue(perf.job_summary(sess, None).empty)

    def test_projection(self):
        p = perf.projection(n_tiles=274_932, n_periods={"monthly": 105, "quarterly": 35},
                            per_core_hour=150, eecu_per_tile={"quarterly": 145, "monthly": 50},
                            cores_per_node=40, nodes=3, quota_eecu_hours=100_000,
                            allocation_core_hours=200_000)
        self.assertEqual(list(p["scope"]), ["monthly only", "quarterly only", "both"])
        both = p[p["scope"] == "both"].iloc[0]
        self.assertEqual(both["tile_periods"], 274_932 * 140)
        self.assertAlmostEqual(both["core_hours"], 274_932 * 140 / 150, places=3)
        self.assertAlmostEqual(both["days_on_3_nodes"], both["core_hours"] / (40 * 24 * 3), places=6)
        quarterly = p[p["scope"] == "quarterly only"].iloc[0]
        self.assertAlmostEqual(quarterly["eecu_hours"], 274_932 * 35 * 145 / 3600, places=3)
        self.assertAlmostEqual(quarterly["months_of_quota"], quarterly["eecu_hours"] / 100_000, places=6)


if __name__ == "__main__":
    unittest.main()
