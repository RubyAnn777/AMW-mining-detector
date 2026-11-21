"""A made-up run with an assembled panel over many periods, mines that persist, and clouds.

    python panel_pipeline/checks/tests/fake_panel.py tests_output/fake_panel   # writes it
    from fake_panel import make_panel_run                                       # from tests

For the analysis notebooks (notebooks/build_tables.py and notebooks/1_... to 4_...), which
need what fake_run.py does not have: several quarters and months, chips that are mines
from some period on, a detector that misses them when the period is cloudy, and AMW's raw
files (yearly 2018 to 2025, quarterly from 2025) as geojson. Two subregions, 24 tiles.
Nothing here means anything; it only has the right shape and some structure to look at.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_run import CHIPS_PER_TILE, DEG, VOTES, QUALITY, chip_ids, tile_key  # noqa: E402

QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]
COLS, ROWS = 6, 4


def tiles() -> pd.DataFrame:
    rows = []
    for ti in range(COLS):
        for tj in range(ROWS):
            for sub in ([1] if ti < 3 else [1, 2] if ti == 3 else [2]):
                rows.append({"subregion": sub, "tile_key": tile_key(ti, tj), "ti": ti, "tj": tj})
    return pd.DataFrame(rows)


def chips(t: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ti, tj in sorted({(a, b) for a, b in zip(t["ti"], t["tj"])}):
        idx = np.arange(CHIPS_PER_TILE)
        col, row = idx % 23, idx // 23
        x0 = -60.0 + ti * DEG + col * (DEG * 24 / 576)
        y0 = -3.0 - tj * DEG - row * (DEG * 24 / 576)
        side = DEG * 48 / 576
        frames.append(pd.DataFrame({
            "chip_id": chip_ids(ti, tj), "tile_key": tile_key(ti, tj), "chip_index": idx.astype(np.int16),
            "minx": x0, "miny": y0 - side, "maxx": x0 + side, "maxy": y0,
            "amw_key_x": np.round(x0 + side / 2, 5), "amw_key_y": np.round(y0 - side / 2, 5),
            "in_region": True, "in_andes": False}))
    return pd.concat(frames, ignore_index=True).sort_values("chip_id").reset_index(drop=True)


def month_index(tag: str) -> int:
    if "Q" in tag:
        y, q = tag.split("Q")
        return int(y) * 12 + (int(q) - 1) * 3
    y, m = tag.split("-")
    return int(y) * 12 + int(m) - 1


def make_panel_run(root: str | Path, amw_dir: str | Path | None = None) -> Path:
    import geopandas as gpd
    import shapely
    import yaml

    root = Path(root)
    amw_dir = Path(amw_dir) if amw_dir else root / "amw"
    for d in ("run", "plan", amw_dir):
        (root / d if isinstance(d, str) else d).mkdir(parents=True, exist_ok=True)
    t = tiles()
    t[["subregion", "tile_key"]].to_csv(root / "plan" / "tiles.csv", index=False)
    c = chips(t)
    c.to_parquet(root / "plan" / "chips.parquet", index=False)
    repo = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((repo / "panel_pipeline" / "config.yaml").read_text())
    cfg["run"] = {"name": root.name, "output_root": str(root.parent)}
    (root / "run" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    rng = np.random.default_rng(11)
    n = len(c)
    lon, lat = ((c["minx"] + c["maxx"]) / 2).to_numpy(), ((c["miny"] + c["maxy"]) / 2).to_numpy()
    # mines: clusters around a few centres, each chip a mine from an onset month on
    centres = rng.uniform([lon.min(), lat.min()], [lon.max(), lat.max()], size=(10, 2))
    onset = np.full(n, 10**9)
    for k, (x, y) in enumerate(centres):
        d = np.hypot(lon - x, lat - y)
        near = d < rng.uniform(0.01, 0.03)
        first = 2019 * 12 + rng.integers(0, 8 * 12) if k < 7 else 2025 * 12 + 9 + rng.integers(0, 6)
        onset[near] = np.minimum(onset[near], first + rng.integers(0, 12, near.sum()))
    sub_of_chip = np.where((c["chip_id"].to_numpy() // 1000 // 10_000 % 1000 - 500) < 3, 1, 2)

    def period_values(tag: str, months: int) -> pd.DataFrame:
        mi = month_index(tag)
        season = 1.0 if (mi % 12) in (0, 1, 2, 3, 11) else 3.0            # wet season: cloudier
        base = rng.gamma(2.0, 1.5 * season * months, n) * np.where(sub_of_chip == 1, 0.6, 1.2)
        n_obs = (base + rng.integers(2, 6, n) * months).astype(np.float32)
        n_clear = np.minimum(base, n_obs).round(2).astype(np.float32)
        empty = np.clip(np.where(n_clear < 1.5, 1 - n_clear / 1.5 + rng.normal(0, 0.1, n), 0), 0, 1).astype(np.float32)
        is_mine = (mi + months - 1) >= onset
        seen = 1 - empty
        mean = np.where(is_mine, 0.75 * seen + 0.05, 0.02) + rng.normal(0, 0.12, n) * np.where(is_mine, 1, 0.15)
        votes = np.clip(mean[:, None] + rng.normal(0, 0.08, (n, 6)), 0, 1)
        fp = rng.uniform(size=n) < 0.002                                   # a few false alarms
        votes[fp] = np.clip(rng.uniform(0.3, 0.8, (fp.sum(), 1)) + rng.normal(0, 0.1, (fp.sum(), 6)), 0, 1)
        votes = votes.astype(np.float32)
        out = pd.DataFrame({"chip_id": c["chip_id"].to_numpy(), "subregion": sub_of_chip.astype(np.int8), "scored": True})
        out["confidence"] = votes.mean(axis=1).astype(np.float64)
        for i, v in enumerate(VOTES):
            out[v] = votes[:, i]
        out["n_obs"], out["n_clear"] = n_obs, n_clear
        out["n_clear_median"] = np.floor(n_clear).astype(np.float32)
        out["empty_share"] = empty
        out["thin_share"] = np.clip(np.where(n_clear < 3, 0.5, 0.05) + rng.normal(0, 0.05, n), 0, 1).astype(np.float32)
        out["clear_day_mean"] = np.where(n_clear > 0, rng.uniform(0, 28 * months, n), np.nan).astype(np.float32)
        out["cs_mean"] = np.clip(0.9 - 0.4 * (n_clear < 2) + rng.normal(0, 0.05, n), 0, 1).astype(np.float32)
        out["cs_clear_mean"] = rng.uniform(0.6, 0.98, n).astype(np.float32)
        out["ndvi_mean"] = np.where(is_mine, 0.2, 0.8).astype(np.float32)
        out["bsi_mean"] = np.where(is_mine, 0.2, -0.3).astype(np.float32)
        out["cirrus_mean"] = rng.uniform(0, 0.03, n).astype(np.float32)
        unscored = rng.uniform(size=n) < 0.001
        out.loc[unscored, ["confidence", *VOTES, *QUALITY]] = np.nan
        out.loc[unscored, "scored"] = False
        return out

    for freq, tags, months in (("quarterly", QUARTERS, 3), ("monthly", MONTHS, 1)):
        for tag in tags:
            layer = period_values(tag, months)
            ok = layer["scored"]
            post = pd.DataFrame({"chip_id": layer["chip_id"]})
            post["det_raw"] = pd.Series(layer["confidence"] > 0.4, dtype="boolean").mask(~ok)
            post["det_andes"] = pd.Series(False, index=post.index, dtype="boolean").mask(~ok)
            for name, t_main in (("loose", 0.43), ("stringent", 0.55)):
                post[f"kth_km_{name}"] = np.where(layer["confidence"] >= t_main, 1.5, np.nan)
                post[f"kth_km_geo_{name}"] = post[f"kth_km_{name}"]
                post[f"det_{name}"] = pd.Series(layer["confidence"] >= t_main, dtype="boolean").mask(~ok)
            for name, table in (("scores", layer), ("postprocess", post)):
                folder = root / "panel" / freq / name / f"period={tag}"
                folder.mkdir(parents=True, exist_ok=True)
                table.to_parquet(folder / "data.parquet", index=False, compression="zstd", row_group_size=3000)
                (folder / "_record.json").write_text(json.dumps({
                    "layer": name, "frequency": freq, "period": tag, "assembled_utc": "2026-11-01T10:00:00+00:00",
                    "counts": {"chips": len(table)},
                    "masked_pixels": {"chips_with_lost_quality_values": 0}}, indent=2))

    # AMW's raw files: yearly 2018-2025 (a mine from its onset year on, missed now and then),
    # quarterly 2025Q1-2026Q3; plus a few AMW-only false alarms
    squares = np.asarray(shapely.box(*(c[k].to_numpy() for k in ("minx", "miny", "maxx", "maxy"))))
    def amw_file(name: str, detected: np.ndarray):
        conf = np.clip(rng.uniform(0.45, 0.95, detected.sum()), 0, 1)
        gpd.GeoDataFrame({"confidence": conf}, geometry=squares[detected], crs="EPSG:4326").to_file(
            amw_dir / f"amazon_basin_{name}_t0.4.geojson", driver="GeoJSON")
    for year in range(2018, 2026):
        # the yearly median picture: a mine that starts late in the year is often missed
        late = (onset // 12 == year) & (onset % 12 >= 8)
        det = (onset // 12 <= year) & (rng.uniform(size=n) < 0.85) & ~(late & (rng.uniform(size=n) < 0.8))
        amw_file(str(year), det | (rng.uniform(size=n) < 0.0005))
    for q in QUARTERS:
        mi = month_index(q)
        det = (onset <= mi + 2) & (rng.uniform(size=n) < 0.8)
        y, k = q.split("Q")
        amw_file(f"Q{k}{y[2:]}", det)
    return root


if __name__ == "__main__":
    print("wrote", make_panel_run(sys.argv[1] if len(sys.argv) > 1 else "tests_output/fake_panel"))
