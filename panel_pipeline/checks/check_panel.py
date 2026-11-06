"""Simple checks of an assembled panel: is it complete, consistent, and what AMW would say?

    python panel_pipeline/checks/check_panel.py --run tests_output/<run>
    python panel_pipeline/checks/check_panel.py --run tests_output/<run> --subregions 2 3
    python panel_pipeline/checks/check_panel.py --run tests_output/<run> --year 2026

Reads every assembled period of both frequencies, a row group at a time, so it needs
little memory even for the whole region. Prints one line per check and period, and a
last line saying whether everything passed. From a notebook:

    from check_panel import check_panel
    report = check_panel(run, subregions=[2, 3])     # a table, one row per check and period

The checks:
  rows          every planned chip once, sorted by chip_id; both layers have the same rows
  scored        every chip has a score, or belongs to a tile listed as failed
  votes         confidence is the mean of the six votes (AMW's ensemble score)
  ranges        scores and shares between 0 and 1, n_clear <= n_obs, no infinite values
  flags         det_raw is confidence > 0.4; stringent within loose within raw;
                flags missing exactly where there is no score
  distances     a distance exactly for the chips in each catalogue (confidence >= t_main);
                the geodesic one within 10% of AMW's
  border        the two scores of every border chip were identical
  months        a quarter's n_obs and n_clear are the sums of its three months'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from periods import Period          # noqa: E402
from read_panel import panel_dir    # noqa: E402

VOTES = [f"vote_{i}" for i in range(1, 7)]
ROWS = 200 * 529                    # the panel's row groups (assemble_period.py)


def tile_number(key: str) -> int:
    """Same as assemble_period.py: a tile key's number, its chip_ids without the last 3 digits."""
    zone, ti, tj = (int(float(part)) for part in key.split(":")[3:])
    return (zone * 1000 + ti + 500) * 10_000 + tj + 5000


def layer_path(base: Path, layer: str, tag: str) -> Path:
    return base / layer / f"period={tag}" / "data.parquet"


def batches(path: Path, columns: list[str]):
    """The file's rows, a row group at a time, as dicts of numpy arrays (missing -> NaN)."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=ROWS, columns=columns):
        yield {c: b.column(c).to_numpy(zero_copy_only=False) for c in columns}


def flag_values(path: Path, column: str):
    """A flag column a row group at a time: (True where set, True where missing)."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=ROWS, columns=[column]):
        a = b.column(column)
        yield (a.fill_null(False).to_numpy(zero_copy_only=False),
               a.is_null().to_numpy(zero_copy_only=False))


def check_period(base: Path, tag: str, n_planned: int) -> list[dict]:
    """All single-period checks of one period. Returns one dict per check."""
    s_path, p_path = layer_path(base, "scores", tag), layer_path(base, "postprocess", tag)
    s_rec = json.loads((s_path.parent / "_record.json").read_text())
    p_rec = json.loads((p_path.parent / "_record.json").read_text())
    th = p_rec["thresholds"]
    have = pq.read_schema(s_path).names
    quality = [c for c in ("n_obs", "n_clear", "empty_share", "thin_share") if c in have]
    s_cols = ["chip_id", "scored", "confidence", *VOTES, *quality]
    p_cols = ["chip_id", "kth_km_loose", "kth_km_geo_loose", "kth_km_stringent",
              "kth_km_geo_stringent"]
    flags = ["det_raw", "det_loose", "det_stringent"]
    failed = np.array([tile_number(k) for k in s_rec["tiles_failed"]], dtype=np.int64)

    n = 0
    last_id = -1
    bad = {k: 0 for k in ("order", "align", "unscored", "votes", "ranges", "inf", "raw",
                          "nested", "missing")}
    extra = {"loose": 0, "stringent": 0}            # a distance outside the catalogue
    absent = {"loose": 0, "stringent": 0}           # a catalogue chip without a distance
    worst_votes, geo_ratio = 0.0, []
    flag_iters = [flag_values(p_path, f) for f in flags]
    for s, p, *f in zip(batches(s_path, s_cols), batches(p_path, p_cols), *flag_iters):
        ids = s["chip_id"]
        n += len(ids)
        bad["order"] += int((np.diff(ids) <= 0).sum()) + int(ids[0] <= last_id)
        last_id = ids[-1]
        bad["align"] += int((ids != p["chip_id"]).sum())
        scored = s["scored"]
        bad["unscored"] += int(np.isin(ids[~scored] // 1000, failed, invert=True).sum())

        conf = s["confidence"]
        votes = np.column_stack([s[v] for v in VOTES]).astype(np.float64)
        gap = np.abs(votes.mean(axis=1) - conf)[scored]
        worst_votes = max(worst_votes, float(gap.max()) if gap.size else 0.0)
        bad["votes"] += int((gap > 1e-6).sum())
        values = np.column_stack([conf, votes])[scored]
        bad["ranges"] += int(((values < 0) | (values > 1)).any(axis=1).sum())
        if "n_clear" in s:
            bad["ranges"] += int((s["n_clear"][scored] > s["n_obs"][scored]).sum())
        for c in ("empty_share", "thin_share"):
            if c in s:
                x = s[c][scored]
                bad["ranges"] += int(((x < 0) | (x > 1)).sum())
        bad["inf"] += int(sum(np.isinf(s[c]).sum() for c in s if s[c].dtype.kind == "f"))
        bad["inf"] += int(sum(np.isinf(p[c]).sum() for c in p_cols[1:]))

        (raw, raw_na), (loose, loose_na), (strict, strict_na) = f
        bad["raw"] += int((raw != (scored & (conf > th["raw"]))).sum())
        bad["nested"] += int((strict & ~loose).sum() + (loose & ~raw).sum())
        bad["missing"] += int(sum((na != ~scored).sum() for na in (raw_na, loose_na, strict_na)))
        for name in ("loose", "stringent"):
            has = np.isfinite(p[f"kth_km_{name}"])
            in_cat = scored & (conf >= th[name]["t_main"])
            extra[name] += int((has & ~in_cat).sum())
            absent[name] += int((in_cat & ~has).sum())
            ok = has & (p[f"kth_km_{name}"] > 0)
            geo_ratio.append(p[f"kth_km_geo_{name}"][ok] / p[f"kth_km_{name}"][ok])
    # Without a distance, by design: a chip dropped as a twin at a zone seam, and every chip
    # of a catalogue too small to have k neighbours (AMW: postprocess.py:213).
    wrong = 0
    for name in ("loose", "stringent"):
        c = p_rec["counts"][name]
        expected = c.get("twins_dropped", 0) + (c["catalogue"] if c["catalogue"] <= th["neighbours_k"] else 0)
        wrong += extra[name] + abs(absent[name] - expected)
    ratio = np.concatenate(geo_ratio) if geo_ratio else np.array([])
    border = s_rec.get("border", {})

    def row(check: str, ok: bool, detail: str) -> dict:
        return {"period": tag, "check": check, "ok": bool(ok), "detail": detail}

    return [
        row("rows", n == n_planned and not bad["order"] and not bad["align"],
            f"{n:,} rows for {n_planned:,} planned chips; {bad['order']} out of order, "
            f"{bad['align']} misaligned"),
        row("scored", not bad["unscored"],
            f"{int(s_rec['scored']):,} scored; {bad['unscored']} unscored outside failed tiles"),
        row("votes", not bad["votes"], f"largest gap {worst_votes:.1e}"),
        row("ranges", not bad["ranges"] and not bad["inf"],
            f"{bad['ranges']} out of range, {bad['inf']} infinite"),
        row("flags", not (bad["raw"] or bad["nested"] or bad["missing"]),
            f"{bad['raw']} raw flags wrong, {bad['nested']} not nested, "
            f"{bad['missing']} missing where they shouldn't be"),
        row("distances", not wrong and (ratio.size == 0 or ((ratio > 0.9) & (ratio < 1.1)).all()),
            f"{wrong} wrong; geodesic / AMW's "
            + (f"{ratio.min():.3f} to {ratio.max():.3f}" if ratio.size else "(no catalogue)")),
        row("border", border.get("copies_differ", 0) == 0,
            f"{border.get('chips_scored_twice', 0):,} chips scored twice, "
            f"{border.get('copies_differ', 0)} copies differ"),
    ]


def check_months(base_m: Path, base_q: Path, quarter: str) -> dict:
    """A quarter's n_obs and n_clear against the sums of its three months'."""
    months = [m.tag for m in Period.parse(quarter).months()]
    files = [layer_path(base_m, "scores", m) for m in months]
    if not all(f.exists() for f in files):
        return {"period": quarter, "check": "months", "ok": True,
                "detail": "not all three months assembled; skipped"}
    worst = {"n_obs": 0.0, "n_clear": 0.0}
    cols = ["chip_id", "scored", "n_obs", "n_clear"]
    for q, *ms in zip(batches(layer_path(base_q, "scores", quarter), cols),
                      *(batches(f, cols) for f in files)):
        both = q["scored"] & np.logical_and.reduce([m["scored"] for m in ms])
        for c in worst:
            total = np.sum([m[c].astype(np.float64) for m in ms], axis=0)
            worst[c] = max(worst[c], float(np.abs(total - q[c])[both].max(initial=0)))
    # per-chip means of whole numbers, stored as 32-bit: equal up to rounding
    ok = worst["n_obs"] < 1e-3 and worst["n_clear"] < 1e-3
    return {"period": quarter, "check": "months", "ok": ok,
            "detail": f"largest gap n_obs {worst['n_obs']:.1e}, n_clear {worst['n_clear']:.1e}"}


def check_panel(run: str | Path, subregions: list[int] | None = None,
                year: int | None = None) -> pd.DataFrame:
    """Every check of every assembled period (or of one year's), as one table."""
    run = Path(run)
    tiles = pd.read_csv(run / "plan" / "tiles.csv")
    chosen = sorted(set(subregions or tiles["subregion"]))
    base = panel_dir(run, subregions)
    n_planned = 529 * tiles.loc[tiles["subregion"].isin(chosen), "tile_key"].nunique()
    rows = []
    for frequency in ("monthly", "quarterly"):
        for rec in sorted((base / frequency / "postprocess").glob("period=*/_record.json")):
            tag = rec.parent.name.removeprefix("period=")
            if year is not None and Period.parse(tag).year != year:
                continue
            rows += [{"frequency": frequency, **r}
                     for r in check_period(base / frequency, tag, n_planned)]
            if frequency == "quarterly":
                rows.append({"frequency": frequency,
                             **check_months(base / "monthly", base / "quarterly", tag)})
    if not rows:
        raise SystemExit(f"CHECK ERROR: nothing assembled in {base}"
                         + (f" for {year}" if year is not None else ""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check an assembled panel.")
    ap.add_argument("--run", required=True, help="the run folder")
    ap.add_argument("--subregions", type=int, nargs="+",
                    help="the panel assembled with --subregions (default: the whole region's)")
    ap.add_argument("--year", type=int, help="only this year's periods, e.g. 2026 (default: all)")
    args = ap.parse_args()
    report = check_panel(args.run, args.subregions, args.year)
    summary = (report.groupby("check", sort=False)
               .agg(periods=("ok", "size"), passed=("ok", "sum"),
                    example=("detail", "first")))
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(f"{report['period'].nunique()} periods checked\n")
        print(summary.to_string())
        failed = report[~report["ok"]]
        if len(failed):
            print("\nFAILED:")
            print(failed.to_string(index=False))
    print(f"\n{'all checks passed' if failed.empty else f'{len(failed)} checks FAILED'}")
    sys.exit(1 if len(failed) else 0)
