"""A small made-up run folder, shaped like a real one, for testing the checks without a run.

    python panel_pipeline/checks/tests/fake_run.py tests_output/fake_run     # writes it
    python -c "from fake_run import make_run; make_run('somewhere')"           # from tests

Two subregions (2 and 3) of 12 tiles, 3 of them on the border between the two (so listed
and scored under both, as plan.py does); months 2025-01 to 2025-03 and the quarter 2025Q1
complete for both, 2025-04 partly done with --limit for subregion 2; the quarter's n_obs and
n_clear are the sums of its months', as in a real run. With `panel=True` the complete periods
are also assembled (scores and postprocess layers, border chips resolved to the higher
score) and a compare.py results folder is written. The scores are random: nothing here
means anything, it only has the right shape, columns and types.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
VOTES = [f"vote_{i}" for i in range(1, 7)]
QUALITY = ["n_obs", "n_clear", "n_clear_median", "empty_share", "thin_share", "clear_day_mean",
           "cs_mean", "cs_clear_mean", "ndvi_mean", "bsi_mean", "cirrus_mean"]
CHIPS_PER_TILE = 529                 # 23 x 23 chips of 48 px, stride 24 px
ZONE = 21
DEG = 0.0518                         # a 576 px tile at 10 m, roughly, in degrees


def tile_key(ti: int, tj: int) -> str:
    return f"552:12:10.0:{ZONE}:{ti}:{tj}"


def chip_ids(ti: int, tj: int) -> np.ndarray:
    base = ((ZONE * 1000 + ti + 500) * 10_000 + tj + 5000) * 1000     # as grid.chip_ids
    return base + np.arange(CHIPS_PER_TILE, dtype=np.int64)


def tiles() -> pd.DataFrame:
    """12 tiles in a 4 x 3 block; column 2 lies on the border and belongs to both subregions."""
    rows = []
    for ti in range(4):
        for tj in range(3):
            for sub in ([2] if ti < 2 else [2, 3] if ti == 2 else [3]):
                rows.append({"subregion": sub, "tile_key": tile_key(ti, tj), "ti": ti, "tj": tj})
    return pd.DataFrame(rows)


def chips_table(t: pd.DataFrame) -> pd.DataFrame:
    """One row per chip of every tile, sorted by chip_id, with plan.py's columns."""
    frames = []
    for ti, tj in sorted({(a, b) for a, b in zip(t["ti"], t["tj"])}):
        idx = np.arange(CHIPS_PER_TILE)
        col, row = idx % 23, idx // 23
        x0 = -60.0 + ti * DEG + col * (DEG * 24 / 576)
        y0 = -3.0 - tj * DEG - row * (DEG * 24 / 576)
        side = DEG * 48 / 576
        frames.append(pd.DataFrame({
            "chip_id": chip_ids(ti, tj), "tile_key": tile_key(ti, tj),
            "chip_index": idx.astype(np.int16),
            "minx": x0, "miny": y0 - side, "maxx": x0 + side, "maxy": y0,
            "amw_key_x": np.round(x0 + side / 2, 5), "amw_key_y": np.round(y0 - side / 2, 5),
            "in_region": True, "in_andes": False}))
    return pd.concat(frames, ignore_index=True).sort_values("chip_id").reset_index(drop=True)


def scores(ids: np.ndarray, seed: int, n_months: int) -> pd.DataFrame:
    """Random scores and quality columns for these chips, detect.py's columns and types."""
    rng = np.random.default_rng(seed)
    votes = np.clip(rng.beta(0.5, 3, size=(len(ids), 6)) + rng.normal(0, 0.03, size=(len(ids), 6)), 0, 1)
    n_obs = rng.integers(4, 8, len(ids)).astype(np.float32) * n_months
    n_clear = np.round(n_obs * rng.uniform(0.1, 0.9, len(ids)), 2).astype(np.float32)
    empty = np.where(n_clear < 1, rng.uniform(0.2, 0.8, len(ids)), 0).astype(np.float32)
    out = pd.DataFrame({"chip_id": ids, "confidence": votes.mean(axis=1).astype(np.float64)})
    for i, v in enumerate(VOTES):
        out[v] = votes[:, i].astype(np.float32)
    out["n_obs"], out["n_clear"] = n_obs, n_clear
    out["n_clear_median"] = np.floor(n_clear).astype(np.float32)
    out["empty_share"], out["thin_share"] = empty, (rng.uniform(0, 0.3, len(ids)) * (empty == 0)).astype(np.float32)
    out["clear_day_mean"] = np.where(n_clear > 0, rng.uniform(0, 28 * n_months, len(ids)), np.nan).astype(np.float32)
    out["cs_mean"] = rng.uniform(0.3, 0.9, len(ids)).astype(np.float32)
    out["cs_clear_mean"] = rng.uniform(0.6, 0.98, len(ids)).astype(np.float32)
    out["ndvi_mean"] = rng.uniform(-0.1, 0.9, len(ids)).astype(np.float32)
    out["bsi_mean"] = rng.uniform(-0.5, 0.5, len(ids)).astype(np.float32)
    out["cirrus_mean"] = rng.uniform(0, 0.05, len(ids)).astype(np.float32)
    return out


def status(frequency: str, period: str, sub: int, planned: list[str], saved: int,
           chips_saved: int, limit: int | None, model: str) -> dict:
    """A _status.json as detect.write_status writes it (the keys the readers use)."""
    return {"run": "fake_run", "frequency": frequency, "period": period, "subregion": sub,
            "ee_project": "fake", "model": model, "model_members": VOTES,
            "complete": saved == len(planned), "tiles_planned": len(planned), "tiles_saved": saved,
            "chips_saved": chips_saved, "tiles_not_tried": len(planned) - saved, "tiles_failed": {},
            "sessions": [{"started_utc": "2026-10-28T12:42:00+00:00",
                          "finished_utc": "2026-10-28T13:40:00+00:00", "limit": limit,
                          "max_workers": 3, "batch_tiles": 500, "flush_minutes": 2.0,
                          "tf_eager": False, "tries": 3, "image_cache": None,
                          "git_commit": "0123456789abcdef", "git_uncommitted_files": []}]}


def make_run(root: str | Path, panel: bool = True, comparison: bool = True) -> Path:
    root = Path(root)
    (root / "run").mkdir(parents=True, exist_ok=True)
    (root / "plan").mkdir(exist_ok=True)
    t = tiles()
    t[["subregion", "tile_key"]].to_csv(root / "plan" / "tiles.csv", index=False)
    chips = chips_table(t)
    chips.to_parquet(root / "plan" / "chips.parquet", index=False)

    cfg = yaml.safe_load((REPO / "panel_pipeline" / "config.yaml").read_text())
    cfg["run"] = {"name": root.name, "output_root": str(root.parent)}
    cfg["periods"] = {"first_month": "2025-01", "last_month": "2025-06",
                      "frequencies": ["monthly", "quarterly"]}
    cfg["region"]["clip_to"] = None
    (root / "run" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = cfg["model"]["path"]
    (root / "run" / "provenance.json").write_text(json.dumps({
        "planned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": "fake",
        "git_commit": "0123456789abcdef", "git_uncommitted_files": [], "python": sys.version.split()[0],
        "packages": {}, "file_sha256": {model: hashlib.sha256(b"fake model").hexdigest()},
        "counts": {"tiles": int(t["tile_key"].nunique()), "chips": len(chips),
                   "in_region": len(chips), "in_andes": 0,
                   "tiles_on_subregion_borders": int((t["tile_key"].value_counts() > 1).sum())}},
        indent=2))

    months = ["2025-01", "2025-02", "2025-03"]
    by_sub = {n: g for n, g in t.groupby("subregion")}
    month_scores: dict[tuple[str, int], pd.DataFrame] = {}
    for sub, g in by_sub.items():
        ids = np.concatenate([chip_ids(a, b) for a, b in zip(g["ti"], g["tj"])])
        keys = list(g["tile_key"])
        for m, month in enumerate(months):
            s = scores(ids, seed=hash((month,)) % 1000, n_months=1)   # same seed for both copies of a border tile
            month_scores[(month, sub)] = s
            folder = root / "detect" / "monthly" / f"period={month}" / f"subregion={sub}"
            folder.mkdir(parents=True, exist_ok=True)
            s.to_parquet(folder / "part_00000.parquet", index=False, compression="zstd")
            (folder / "_status.json").write_text(json.dumps(
                status("monthly", month, sub, keys, len(keys), len(s), None, model), indent=2))
        # the quarter: the months' counts add up; the picture (scores) is its own
        q = scores(ids, seed=hash(("2025Q1",)) % 1000, n_months=3)
        q["n_obs"] = sum(month_scores[(m, sub)]["n_obs"] for m in months).astype(np.float32)
        q["n_clear"] = sum(month_scores[(m, sub)]["n_clear"] for m in months).astype(np.float32)
        folder = root / "detect" / "quarterly" / "period=2025Q1" / f"subregion={sub}"
        folder.mkdir(parents=True, exist_ok=True)
        q.to_parquet(folder / "part_00000.parquet", index=False, compression="zstd")
        (folder / "_status.json").write_text(json.dumps(
            status("quarterly", "2025Q1", sub, keys, len(keys), len(q), None, model), indent=2))
        month_scores[("2025Q1", sub)] = q
    # 2025-04, subregion 2, partly done with --limit 2: two tiles in two part files
    g = by_sub[2]
    for i, (a, b) in enumerate(list(zip(g["ti"], g["tj"]))[:2]):
        folder = root / "detect" / "monthly" / "period=2025-04" / "subregion=2"
        folder.mkdir(parents=True, exist_ok=True)
        scores(chip_ids(a, b), seed=40 + i, n_months=1).to_parquet(
            folder / f"part_{i:05d}.parquet", index=False, compression="zstd")
    (folder / "_status.json").write_text(json.dumps(
        status("monthly", "2025-04", 2, list(g["tile_key"]), 2, 2 * CHIPS_PER_TILE, 2, model), indent=2))
    (folder / "_log.txt").write_text(
        "2026-10-28T12:42:00+00:00  2025-04  starting: 2 tiles to score, 3 workers, a part every 500 tiles or 2 min\n")

    if panel:
        for freq, tag in (("monthly", "2025-01"), ("monthly", "2025-02"), ("monthly", "2025-03"),
                          ("quarterly", "2025Q1")):
            both = pd.concat([month_scores[(tag, n)].assign(subregion=np.int8(n)) for n in (2, 3)])
            # border chips: the copy with the higher confidence, as AMW's dedupe does
            one = (both.sort_values(["chip_id", "confidence"], ascending=[True, False])
                       .drop_duplicates("chip_id").set_index("chip_id").reindex(chips["chip_id"]).reset_index())
            one.insert(2, "scored", one["confidence"].notna())
            layer = one[["chip_id", "subregion", "scored", "confidence", *VOTES, *QUALITY]]
            post = pd.DataFrame({"chip_id": layer["chip_id"]})
            det = pd.Series(layer["confidence"] > 0.4, dtype="boolean").mask(~layer["scored"])
            post["det_raw"] = det
            post["det_andes"] = pd.Series(False, index=post.index, dtype="boolean").mask(~layer["scored"])
            for name, t_main in (("loose", 0.43), ("stringent", 0.55)):
                post[f"kth_km_{name}"] = np.where(layer["confidence"] >= t_main, 1.5, np.nan)
                post[f"kth_km_geo_{name}"] = post[f"kth_km_{name}"]
                post[f"det_{name}"] = pd.Series(layer["confidence"] >= t_main, dtype="boolean").mask(~layer["scored"])
            for name, table in (("scores", layer), ("postprocess", post)):
                folder = root / "panel" / freq / name / f"period={tag}"
                folder.mkdir(parents=True, exist_ok=True)
                table.to_parquet(folder / "data.parquet", index=False, compression="zstd")
                (folder / "_record.json").write_text(json.dumps({
                    "layer": name, "frequency": freq, "period": tag, "assembled_utc": "2026-11-01T10:00:00+00:00",
                    "inputs": {str(n): {"tiles": int(len(by_sub[n]))} for n in (2, 3)},
                    "counts": {"chips": len(table), "scored": int(layer["scored"].sum()),
                               "raw": int(post["det_raw"].sum())}}, indent=2))

    if comparison:
        out = root / "amw_comparison"
        out.mkdir(exist_ok=True)
        q = pd.concat([month_scores[("2025Q1", n)].assign(subregion=n) for n in (2, 3)])
        q = q.sort_values(["chip_id", "confidence"], ascending=[True, False]).drop_duplicates("chip_id")
        rng = np.random.default_rng(7)
        q["amw_confidence"] = np.where(rng.uniform(size=len(q)) < 0.9, q["confidence"], np.nan)
        for m in months:
            q[f"conf_{m}"] = q["chip_id"].map(pd.concat([month_scores[(m, n)] for n in (2, 3)])
                                               .drop_duplicates("chip_id").set_index("chip_id")["confidence"])
        q["months_detected"] = (q[[f"conf_{m}" for m in months]] > 0.4).sum(axis=1)
        q["months_scored"] = 3
        kept = q[(q["confidence"] > 0.4) | (q["amw_confidence"] > 0.4) | (q["months_detected"] > 0)]
        kept[["chip_id", "subregion", "confidence", "amw_confidence", *[f"conf_{m}" for m in months],
              "months_detected", "months_scored"]].to_parquet(out / "chips_2025Q1.parquet", index=False)
        pd.DataFrame([{"quarter": "2025Q1", "subregion": n, "AMW rows on our chips": int(kept.loc[kept["subregion"] == n, "amw_confidence"].notna().sum()),
                       "should have matched": 0, "chips sharing a key": 0} for n in (2, 3)]).to_csv(out / "joins.csv", index=False)
        pd.DataFrame([{"frequency": "quarterly", "period": "2025Q1", "subregion": n, "state": "complete",
                       "tiles": f"{len(by_sub[n])}/{len(by_sub[n])}", "failed": 0} for n in (2, 3)]).to_csv(out / "inputs.csv", index=False)
        pd.DataFrame([{"quarter": "2025Q1", "file": "amazon_basin_Q125_t0.4.geojson", "rows": 123, "sha256": "fake"}]).to_csv(out / "amw_files.csv", index=False)
    return root


if __name__ == "__main__":
    print("wrote", make_run(sys.argv[1] if len(sys.argv) > 1 else "tests_output/fake_run"))
