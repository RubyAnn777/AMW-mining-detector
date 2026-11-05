"""Sanity check of a run's scores before assembly: structure, values, what broke.

    python panel_pipeline/checks/check_detect.py --run /work/.../aca_full_2026-11-01 --year 2026
    ... --sample 0.2      read the value columns of every 5th row group only (faster)

Reads detect/<frequency>/period=<tag>/subregion=<n>/ directly, so it works on periods
that are still running or have failed tiles. It changes nothing. Prints:

  1. structure  per task: tiles and chips saved vs planned, failed tiles, duplicate chips,
                chips per tile, and whether a subregion has the same chips in every
                complete period (the panel needs every chip in every period)
  2. values     per period: share of chips scored above 0.4 / 0.55, confidence vs votes,
                quality columns (images, clear images, empty share, NDVI) and anything
                missing, infinite or out of range
  3. months     per subregion: a quarter's total n_obs and n_clear against the sum of
                its three months (they should match when all four are complete)
  4. verdict    a short list of what looks wrong
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

VOTES = [f"vote_{i}" for i in range(1, 7)]
QUALITY = ["n_obs", "n_clear", "empty_share", "thin_share", "clear_day_mean",
           "cs_mean", "cs_clear_mean", "ndvi_mean", "bsi_mean", "cirrus_mean"]
SHARES = {"empty_share", "thin_share", "cs_mean", "cs_clear_mean"}
INDICES = {"ndvi_mean", "bsi_mean"}            # between -1 and 1
MOD = (1 << 61) - 1                            # for the chip-set fingerprint


def task_folders(run: Path, year: str | None):
    for freq in ("monthly", "quarterly"):
        for pdir in sorted((run / "detect" / freq).glob("period=*")):
            tag = pdir.name.removeprefix("period=")
            if year and not tag.startswith(year):
                continue
            for sdir in sorted(pdir.glob("subregion=*"),
                               key=lambda p: int(p.name.split("=")[1])):
                yield freq, tag, int(sdir.name.split("=")[1]), sdir


class Stats:
    """Running sums for one column: count, missing, infinite, min, max, mean, out of range."""

    def __init__(self):
        self.n = self.na = self.inf = self.bad = 0
        self.total = 0.0
        self.lo, self.hi = np.inf, -np.inf

    def add(self, x: np.ndarray, lo: float | None = None, hi: float | None = None):
        x = x.astype(np.float64)
        self.n += len(x)
        self.na += int(np.isnan(x).sum())
        self.inf += int(np.isinf(x).sum())
        ok = x[np.isfinite(x)]
        if ok.size:
            self.total += float(ok.sum())
            self.lo, self.hi = min(self.lo, ok.min()), max(self.hi, ok.max())
            if lo is not None:
                self.bad += int(((ok < lo) | (ok > hi)).sum())

    def merge(self, o: "Stats"):
        self.n += o.n; self.na += o.na; self.inf += o.inf; self.bad += o.bad
        self.total += o.total; self.lo = min(self.lo, o.lo); self.hi = max(self.hi, o.hi)

    @property
    def mean(self):
        ok = self.n - self.na - self.inf
        return self.total / ok if ok else float("nan")


def read_task(folder: Path, sample: float) -> dict:
    status = json.loads((folder / "_status.json").read_text()) if (folder / "_status.json").exists() else {}
    files = sorted(folder.glob("part_*.parquet"))
    out = {"status": status, "files": len(files), "rows": 0, "dups": 0, "tiles": 0,
           "per_tile": (0, 0), "fp": (0, 0), "cols": defaultdict(Stats), "above": {},
           "gap": 0.0, "nclear_gt_nobs": 0, "zero_clear": 0, "sampled": 0,
           "missing_cols": [], "sum_obs": 0.0, "sum_clear": 0.0, "error": None}
    if not files:
        return out
    try:
        ids = np.concatenate([pq.read_table(f, columns=["chip_id"])["chip_id"].to_numpy()
                              for f in files]).astype(np.int64)
        out["rows"] = len(ids)
        uniq = np.unique(ids)
        out["dups"] = len(ids) - len(uniq)
        _, per_tile = np.unique(uniq // 1000, return_counts=True)
        out["tiles"] = len(per_tile)
        out["per_tile"] = (int(per_tile.min()), int(per_tile.max()))
        out["fp"] = (len(uniq), int(np.bitwise_xor.reduce((uniq * 2654435761) & MOD)))
        names = pq.read_schema(files[0]).names
        out["missing_cols"] = [c for c in ["confidence", *VOTES, *QUALITY] if c not in names]
        cols = [c for c in ["confidence", *VOTES, *QUALITY] if c in names]
        step = max(1, round(1 / sample))
        k = 0
        above = {0.2: 0, 0.4: 0, 0.55: 0, 0.8: 0}
        for f in files:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                need_all = {"n_obs", "n_clear"} & set(cols)      # totals for the months check
                if k % step:
                    k += 1
                    if need_all:
                        t = pf.read_row_group(g, columns=sorted(need_all))
                        out["sum_obs"] += float(np.nansum(t["n_obs"].to_numpy(zero_copy_only=False)))
                        out["sum_clear"] += float(np.nansum(t["n_clear"].to_numpy(zero_copy_only=False)))
                    continue
                k += 1
                t = pf.read_row_group(g, columns=cols)
                a = {c: t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in cols}
                out["sampled"] += t.num_rows
                conf = a["confidence"]
                out["cols"]["confidence"].add(conf, 0, 1)
                for th in above:
                    above[th] += int((conf > th).sum())
                if all(v in a for v in VOTES):
                    votes = np.column_stack([a[v] for v in VOTES])
                    for v in VOTES:
                        out["cols"][v].add(a[v], 0, 1)
                    gap = np.abs(votes.mean(axis=1) - conf)
                    out["gap"] = max(out["gap"], float(np.nanmax(gap)) if gap.size else 0.0)
                for c in QUALITY:
                    if c not in a:
                        continue
                    lo, hi = (0, 1) if c in SHARES else (-1, 1) if c in INDICES else (None, None)
                    if c in ("n_obs", "n_clear", "clear_day_mean"):
                        lo, hi = 0, 1e6
                    out["cols"][c].add(a[c], lo, hi)
                if "n_obs" in a and "n_clear" in a:
                    out["nclear_gt_nobs"] += int((a["n_clear"] > a["n_obs"] + 1e-4).sum())
                    out["zero_clear"] += int((a["n_clear"] == 0).sum())
                    out["sum_obs"] += float(np.nansum(a["n_obs"]))
                    out["sum_clear"] += float(np.nansum(a["n_clear"]))
        out["above"] = above
    except Exception as e:                       # a broken file is a finding, not a crash
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--year", default=None, help="only periods of this year, e.g. 2026")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="share of row groups whose value columns are read (default all)")
    args = ap.parse_args()
    run = Path(args.run)
    problems = []
    res = {}
    print(f"run {run}" + (f", periods of {args.year}" if args.year else "")
          + (f", values from ~{args.sample:.0%} of rows" if args.sample < 1 else ""))

    print("\n== 1. structure: one line per task ==")
    print(f"{'freq':9} {'period':7} {'sub':>3}  {'state':9} {'tiles saved/planned':>21} "
          f"{'failed':>6} {'chips':>12} {'dups':>5} {'chips/tile':>10}")
    for freq, tag, sub, folder in task_folders(run, args.year):
        r = read_task(folder, args.sample)
        res[(freq, tag, sub)] = r
        s = r["status"]
        state = ("ERROR" if r["error"] else "complete" if s.get("complete") else
                 "no status" if not s else "partial")
        failed = len(s.get("tiles_failed", {}))
        pt = f"{r['per_tile'][0]}-{r['per_tile'][1]}" if r["per_tile"][0] != r["per_tile"][1] else str(r["per_tile"][0])
        print(f"{freq:9} {tag:7} {sub:>3}  {state:9} "
              f"{s.get('tiles_saved', 0):>10,}/{s.get('tiles_planned', 0):<10,} {failed:>6} "
              f"{r['rows']:>12,} {r['dups']:>5} {pt:>10}")
        where = f"{freq} {tag} subregion {sub}"
        if r["error"]:
            problems.append(f"{where}: cannot read: {r['error']}")
        if r["dups"]:
            problems.append(f"{where}: {r['dups']:,} chips saved twice")
        if s and r["tiles"] != s.get("tiles_saved"):
            problems.append(f"{where}: files hold {r['tiles']:,} tiles, _status.json says {s.get('tiles_saved'):,}")
        if s and r["rows"] - r["dups"] != s.get("chips_saved", r["rows"]):
            problems.append(f"{where}: files hold {r['rows'] - r['dups']:,} chips, _status.json says {s.get('chips_saved'):,}")
        if r["per_tile"][0] != r["per_tile"][1]:
            problems.append(f"{where}: tiles hold {pt} chips (expected the same number each)")
        if failed:
            problems.append(f"{where}: {failed} failed tile(s), e.g. "
                            f"{next(iter(s['tiles_failed'].values()))[:100]}")
        if r["missing_cols"]:
            problems.append(f"{where}: columns missing: {', '.join(r['missing_cols'])}")

    # Same chips in every complete period of a subregion (monthly and quarterly alike)?
    by_sub = defaultdict(set)
    for (freq, tag, sub), r in res.items():
        if r["status"].get("complete") and not r["error"]:
            by_sub[sub].add(r["fp"])
    print("\nsame chips in every complete period, per subregion:")
    for sub in sorted(by_sub):
        fps = by_sub[sub]
        n = next(iter(fps))[0]
        ok = len(fps) == 1
        print(f"  subregion {sub}: {'yes' if ok else 'NO'}, {n:,} chips"
              + ("" if ok else f" ({len(fps)} different chip sets)"))
        if not ok:
            problems.append(f"subregion {sub}: complete periods hold different chip sets")

    print("\n== 2. values: one line per period (all subregions together) ==")
    print(f"{'freq':9} {'period':7} {'chips read':>12} {'>0.2':>6} {'>0.4':>6} {'>0.55':>6} "
          f"{'>0.8':>6} {'conf mean':>9} {'n_obs':>6} {'n_clear':>7} {'no clear':>8} "
          f"{'empty':>6} {'ndvi':>5} {'NaN conf':>8}")
    periods = sorted({(f, t) for f, t, _ in res}, key=lambda x: (x[0], x[1]))
    for freq, tag in periods:
        rs = [r for (f, t, _), r in res.items() if (f, t) == (freq, tag) and r["sampled"]]
        if not rs:
            continue
        cols = defaultdict(Stats)
        for r in rs:
            for c, st in r["cols"].items():
                cols[c].merge(st)
        n = sum(r["sampled"] for r in rs)
        ab = {th: sum(r["above"].get(th, 0) for r in rs) for th in (0.2, 0.4, 0.55, 0.8)}
        zc = sum(r["zero_clear"] for r in rs)
        g = lambda c: cols[c].mean if c in cols else float("nan")
        print(f"{freq:9} {tag:7} {n:>12,} {pct(ab[0.2], n)} {pct(ab[0.4], n)} {pct(ab[0.55], n)} "
              f"{pct(ab[0.8], n)} {g('confidence'):9.4f} {g('n_obs'):6.1f} {g('n_clear'):7.1f} "
              f"{pct(zc, n):>8} {g('empty_share'):6.3f} {g('ndvi_mean'):5.2f} "
              f"{pct(cols['confidence'].na, n):>8}")
        where = f"{freq} {tag}"
        for c, st in cols.items():
            if st.inf:
                problems.append(f"{where}: {c} infinite in {st.inf:,} chips")
            if st.bad:
                problems.append(f"{where}: {c} out of range in {st.bad:,} chips ({st.lo:.3g}..{st.hi:.3g})")
            if c != "confidence" and c not in VOTES and st.na == st.n and st.n:
                problems.append(f"{where}: {c} missing in every chip")
        if cols["confidence"].na:
            problems.append(f"{where}: confidence missing in {cols['confidence'].na:,} chips")
        gap = max(r["gap"] for r in rs)
        if gap > 1e-5:
            problems.append(f"{where}: confidence differs from the mean of the votes by up to {gap:.2g}")
        bad = sum(r["nclear_gt_nobs"] for r in rs)
        if bad:
            problems.append(f"{where}: n_clear > n_obs in {bad:,} chips")
        if "empty_share" in cols and cols["empty_share"].hi == 0:
            problems.append(f"{where}: empty_share is 0 everywhere (the pre-1-Nov masking bug?)")

    print("\n== 3. months vs quarter: total images, quarter / sum of its 3 months ==")
    for (freq, tag, sub), r in sorted(res.items()):
        if freq != "quarterly" or not r["status"].get("complete"):
            continue
        y, q = int(tag[:4]), int(tag[-1])
        months = [res.get(("monthly", f"{y}-{3 * (q - 1) + i:02d}", sub)) for i in (1, 2, 3)]
        if not all(m and m["status"].get("complete") for m in months):
            print(f"  {tag} subregion {sub}: a month is not complete yet, skipped")
            continue
        ro = r["sum_obs"] / sum(m["sum_obs"] for m in months)
        rc = r["sum_clear"] / sum(m["sum_clear"] for m in months)
        print(f"  {tag} subregion {sub}: n_obs {ro:.4f}   n_clear {rc:.4f}")
        if abs(ro - 1) > 0.01 or abs(rc - 1) > 0.01:
            problems.append(f"{tag} subregion {sub}: quarter's images are not the sum of its months' "
                            f"(n_obs x{ro:.3f}, n_clear x{rc:.3f})")

    print("\n== 4. verdict ==")
    if not problems:
        print("nothing looks wrong")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    main()
