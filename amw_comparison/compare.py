"""Line our detect.py output up with AMW's published detections, chip by chip.

Used by compare.ipynb and tests/. Reads only; writes nothing.

The join: AMW's files carry no chip ID, only each chip's square. AMW recognises a
chip by its rounded centre (centroid_key, core/postprocess.py:113), and plan.py
stored that same key for every chip (amw_key_x/y in plan/chips.parquet). So an AMW
row is one of our chips exactly when the two keys are equal.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.append(str(REPO / "core"))
from postprocess import centroid_key  # noqa: E402  (AMW code)

AMW_DIR = HERE / "data" / "amw"      # AMW's files, as you downloaded them (ignored by git)
AMW_PANEL = AMW_DIR / "amazon_basin_detections.geojson"   # one row per location (MANIFEST.yaml)
AMW_KINDS = {"raw": "t0.4", "postprocessed": "t0.43_t-iso0.75"}   # MANIFEST.yaml filename_grammar
RAW_THRESHOLD = 0.4                  # AMW keeps confidence > 0.4, strictly (inference_engine.py:803)
OUTCOMES = ["both", "only ours", "only AMW", "neither"]
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ------------------------------------------------------------ reading

def amw_tag(quarter: str) -> str:
    """Our quarter tag to AMW's: '2026Q3' -> 'Q326' (pipeline_config.py:53)."""
    year, q = quarter.split("Q")
    return f"Q{q}{year[2:]}"


def amw_file(quarter: str, kind: str = "raw", folder: Path = AMW_DIR) -> Path:
    """AMW's file for a quarter: raw detections, or after AMW's postprocess filter."""
    return folder / f"amazon_basin_{amw_tag(quarter)}_{AMW_KINDS[kind]}.geojson"


def read_amw(path: Path) -> pd.DataFrame:
    """AMW's published raw detections: one row per chip, with its location key."""
    gdf = gpd.read_file(path)
    key_x, key_y = centroid_key(gdf)               # AMW's key, as AMW computes it
    out = pd.DataFrame({"amw_key_x": key_x, "amw_key_y": key_y,
                        "amw_confidence": gdf["confidence"].to_numpy(dtype=np.float64)})
    if "preds" in gdf.columns:                     # the six votes (parse_votes)
        out["amw_preds"] = gdf["preds"].to_numpy()
    return out


def task_folder(run: Path, period: str, subregion: int, frequency: str = "quarterly") -> Path:
    """One detect.py task's folder; period is '2025Q1' or, for monthly, '2025-01'."""
    return run / "detect" / frequency / f"period={period}" / f"subregion={subregion}"


def read_ours(run: Path, period: str, subregion: int, votes: bool = False,
              only=None, frequency: str = "quarterly",
              cols=("chip_id", "confidence", "n_obs", "n_clear", "empty_share")) -> pd.DataFrame:
    """Our scores for one period and subregion: every chip, above threshold or not.

    Reads every part file in the folder (part_00000.parquet once merged, or
    part_00001.parquet ... while a period is unfinished or was never merged).
    votes: add vote_1..6. only: read just these chip_ids (saves memory).
    """
    cols = list(cols)
    if votes:
        cols += [f"vote_{i}" for i in range(1, 7)]
    folder = task_folder(run, period, subregion, frequency)
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part files in {folder}; has detect.py run?")
    filters = None if only is None else [("chip_id", "in", list(map(int, only)))]
    return pd.concat([pd.read_parquet(p, columns=cols, filters=filters) for p in parts],
                     ignore_index=True)


def read_status(run: Path, period: str, subregion: int, frequency: str = "quarterly") -> dict:
    return json.loads((task_folder(run, period, subregion, frequency) / "_status.json").read_text())


def read_chips(run: Path, subregion: int | None = None) -> pd.DataFrame:
    """Our planned chips (of one subregion's tiles, if given).

    Without tile_key: as text it would take most of the memory. For one
    subregion the file is filtered while it is read, so a whole-basin plan
    (about 146 M chips) never sits in memory at once.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    cols = ["chip_id", "minx", "miny", "maxx", "maxy", "amw_key_x", "amw_key_y"]
    data = ds.dataset(run / "plan" / "chips.parquet", format="parquet")
    keep = None
    if subregion is not None:
        tiles = pd.read_csv(run / "plan" / "tiles.csv")
        keys = tiles.loc[tiles["subregion"] == subregion, "tile_key"]
        numbers = pa.array([key_number(k) for k in keys], type=pa.int64())
        keep = pc.is_in(pc.divide(pc.field("chip_id"), 1000), value_set=numbers)
    chips = data.to_table(columns=cols, filter=keep).to_pandas()
    chips["tile"] = chips["chip_id"] // 1000      # the tile's number (grid.py, chip_ids)
    return chips


def frozen_config(run: Path) -> dict:
    """The run's own settings, as plan.py copied them."""
    return yaml.safe_load((run / "run" / "config.yaml").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ joining

def match(chips: pd.DataFrame, amw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach AMW's rows to our chips by AMW's key.

    Returns our chips with amw_confidence (NaN where AMW has no row), and the AMW
    rows that are none of our chips.
    """
    if chips.duplicated(["amw_key_x", "amw_key_y"]).any():
        raise ValueError("two of our chips share an AMW key; the join would be ambiguous")
    if amw.duplicated(["amw_key_x", "amw_key_y"]).any():
        raise ValueError("AMW's file lists a location twice; was it deduplicated?")
    joined = chips.merge(amw, on=["amw_key_x", "amw_key_y"], how="left",
                         validate="one_to_one")
    ours = pd.MultiIndex.from_frame(chips[["amw_key_x", "amw_key_y"]])
    theirs = pd.MultiIndex.from_frame(amw[["amw_key_x", "amw_key_y"]])
    return joined, amw[~theirs.isin(ours)]


def should_have_matched(unmatched: pd.DataFrame, chips: pd.DataFrame) -> pd.Series:
    """AMW rows centred among our chip centres that still matched none of them.

    Each tile's chip centres form a regular grid; an AMW centre inside that grid
    (to within half a step) must be one of its points. Anything counted here
    means the grids differ. Rows just outside, e.g. a neighbouring tile's chips
    hanging over our edge, are not counted.
    """
    if unmatched.empty:
        return pd.Series(False, index=unmatched.index)
    half_step = (chips["maxx"] - chips["minx"]).median() / 4   # chip 48 px, step 24 px
    zones = chips.groupby("tile").agg(x0=("amw_key_x", "min"), x1=("amw_key_x", "max"),
                                      y0=("amw_key_y", "min"), y1=("amw_key_y", "max"))
    boxes = shapely.box(zones["x0"] - half_step, zones["y0"] - half_step,
                        zones["x1"] + half_step, zones["y1"] + half_step)
    points = shapely.points(unmatched["amw_key_x"], unmatched["amw_key_y"])
    hit, _ = shapely.STRtree(boxes).query(points, predicate="within")
    return pd.Series(np.isin(np.arange(len(unmatched)), hit), index=unmatched.index)


def outcome(table: pd.DataFrame, threshold: float = RAW_THRESHOLD) -> pd.Series:
    """Per chip: 'both', 'only ours', 'only AMW' or 'neither' detected it.

    AMW's file has only chips above its threshold, so a missing AMW row means
    AMW scored the chip at or below it.
    """
    ours = table["confidence"] > threshold
    amw = table["amw_confidence"] > threshold     # NaN > x is False
    labels = np.select([ours & amw, ours & ~amw, ~ours & amw], OUTCOMES[:3], OUTCOMES[3])
    return pd.Series(labels, index=table.index, name="outcome")


def agreement(table: pd.DataFrame, threshold: float = RAW_THRESHOLD) -> pd.Series:
    """The 2×2 counts, and the overlap of the two detection sets."""
    counts = outcome(table, threshold).value_counts().reindex(OUTCOMES, fill_value=0)
    either = counts["both"] + counts["only ours"] + counts["only AMW"]
    return pd.concat([counts, pd.Series({
        "ours total": counts["both"] + counts["only ours"],
        "AMW total": counts["both"] + counts["only AMW"],
        # chips both detected / chips either detected (1 = the same set)
        "overlap (Jaccard)": counts["both"] / either if either else np.nan,
    })])


def score_gap(table: pd.DataFrame) -> pd.Series:
    """How close the two scores are, over chips both sides detected."""
    both = table[outcome(table) == "both"]
    gap = (both["confidence"] - both["amw_confidence"]).abs()
    return pd.Series({
        "chips": len(both),
        "identical (gap <= 1e-6)": (gap <= 1e-6).mean(),
        "gap <= 0.001": (gap <= 1e-3).mean(),
        "gap <= 0.01": (gap <= 1e-2).mean(),
        "median gap": gap.median(),
        "largest gap": gap.max(),
    })


def parse_votes(preds: pd.Series) -> np.ndarray:
    """AMW's 'preds' column (the six member votes) as an (n, 6) array.

    The published files hold numpy's printout, '[0.0899 0.3551 ...]', sometimes
    wrapped over two lines; inference_engine.py:830 would write '[0.0899, ...]'.
    Reading every number in the text handles both.
    """
    return np.array([[float(x) for x in NUMBER.findall(p)] for p in preds],
                    dtype=np.float64)


def border_tiles(run: Path, subregion: int) -> set[int]:
    """Numbers of our tiles that AMW also scored in a neighbouring subregion's run.

    Such a chip appears twice before AMW's concat, which keeps the higher
    confidence (postprocess.py:146), so AMW's number may come from the other run.
    """
    from descarteslabs.geo import DLTile
    pattern = frozen_config(run)["region"]["subregion_files"]
    other = shapely.union_all([gpd.read_file(REPO / pattern.format(n=n)).union_all()
                               for n in range(1, 7) if n != subregion])
    shapely.prepare(other)
    tiles = pd.read_csv(run / "plan" / "tiles.csv")
    keys = tiles.loc[tiles["subregion"] == subregion, "tile_key"]
    return {tile_number(t) for t in map(DLTile.from_key, keys)
            if other.intersects(t.geometry)}


def tile_number(tile) -> int:
    """Same number as chip_id // 1000 (grid.py, chip_ids)."""
    return (tile.zone * 1000 + tile.ti + 500) * 10_000 + tile.tj + 5000


def key_number(tile_key: str) -> int:
    """tile_number from a tile key, '552:12:10.0:21:60:-100' = size:pad:res:zone:column:row."""
    zone, column, row = (int(v) for v in tile_key.split(":")[3:])
    return (zone * 1000 + column + 500) * 10_000 + row + 5000
