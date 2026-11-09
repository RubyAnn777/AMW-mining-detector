"""Reading pieces of an assembled panel for the notebooks, without loading a whole period.

A panel folder holds, per frequency and period, two layer files with the same rows in the
same order (every planned chip once, sorted by chip_id):

    <panel>/<frequency>/<layer>/period=<tag>/data.parquet     layer = scores | postprocess
    <panel>/<frequency>/<layer>/period=<tag>/_record.json     written last: the period is done

A basin-wide period has 145 million rows, so everything here reads single columns, a few
row groups, or rows filtered by chip_id.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

LAYERS = ("scores", "postprocess")
FREQUENCIES = ("monthly", "quarterly")
PLAN_COLUMNS = ["chip_id", "tile_key", "chip_index", "minx", "miny", "maxx", "maxy", "in_region", "in_andes"]


def panels(run: Path) -> list[str]:
    """The panel folders of a run: "panel" (the whole region) and "panel_subregions_..."."""
    return sorted(p.name for p in Path(run).glob("panel*") if p.is_dir())


def layer_path(panel: Path, frequency: str, layer: str, tag: str) -> Path:
    return Path(panel) / frequency / layer / f"period={tag}" / "data.parquet"


def tags(panel: Path, frequency: str) -> list[str]:
    """The periods of one frequency whose two layers are both written."""
    base = Path(panel) / frequency / "postprocess"
    return sorted(p.name.removeprefix("period=") for p in base.glob("period=*")
                  if (p / "_record.json").exists()
                  and (Path(panel) / frequency / "scores" / p.name / "_record.json").exists())


def records(panel: Path) -> list[Path]:
    """Every _record.json of the panel: a newer one means the panel changed."""
    return sorted(Path(panel).glob("*/*/period=*/_record.json"))


def record(panel: Path, frequency: str, layer: str, tag: str) -> dict:
    return json.loads((Path(panel) / frequency / layer / f"period={tag}" / "_record.json").read_text())


def column(panel: Path, frequency: str, layer: str, tag: str, name: str) -> np.ndarray:
    """One column of one period as a numpy array. A missing flag reads as False here;
    the `scored` column tells "no score" and "not detected" apart."""
    a = pq.read_table(layer_path(panel, frequency, layer, tag), columns=[name]).column(name)
    if a.type == pa.bool_():
        a = a.fill_null(False)
    return a.to_numpy()


def rows_at(path: Path, positions: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """Rows at these positions of one layer file, reading only the row groups that hold them."""
    f = pq.ParquetFile(path)
    starts = np.concatenate([[0], np.cumsum([f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)])])
    positions = np.unique(np.asarray(positions, dtype=np.int64))
    groups = np.searchsorted(starts, positions, side="right") - 1
    parts = [f.read_row_group(int(g), columns=columns).to_pandas()
             .iloc[positions[groups == g] - starts[g]].set_axis(positions[groups == g])
             for g in np.unique(groups)]
    return pd.concat(parts) if parts else pd.DataFrame(columns=columns)


def period_rows(panel: Path, frequency: str, tag: str, positions: np.ndarray,
                columns: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Both layers of one period at these row positions, side by side, with `period` added."""
    layers = []
    for layer in LAYERS:
        cols = None if columns is None else ["chip_id", *columns.get(layer, [])]
        layers.append(rows_at(layer_path(panel, frequency, layer, tag), positions, cols))
    if not layers[0]["chip_id"].equals(layers[1]["chip_id"]):
        raise ValueError(f"{frequency} {tag}: the two layers' rows don't line up")
    out = pd.concat([layers[0], layers[1].drop(columns="chip_id")], axis=1)
    out.insert(1, "period", tag)
    return out


def rows_of_chips(panel: Path, frequency: str, tag: str, chip_ids, columns: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Both layers of one period for these chip IDs. The files are sorted by chip_id, so
    the row-group statistics let Arrow skip almost all of a period."""
    ids = pa.array(np.asarray(list(chip_ids), dtype=np.int64))
    parts = []
    for layer in LAYERS:
        path = layer_path(panel, frequency, layer, tag)
        names = pq.read_schema(path).names
        cols = names if columns is None else ["chip_id", *[c for c in columns.get(layer, []) if c in names]]
        parts.append(ds.dataset(path).to_table(columns=cols, filter=pc.field("chip_id").isin(ids)).to_pandas()
                     .set_index("chip_id"))
    out = parts[0].join(parts[1], how="inner").reset_index()
    out.insert(1, "period", tag)
    return out


def chip_facts(run: Path, chip_ids) -> pd.DataFrame:
    """What never changes about these chips, from the plan, with the square's centre."""
    t = ds.dataset(Path(run) / "plan" / "chips.parquet").to_table(
        columns=PLAN_COLUMNS, filter=pc.field("chip_id").isin(pa.array(np.asarray(list(chip_ids), dtype=np.int64)))).to_pandas()
    t["lon"], t["lat"] = (t["minx"] + t["maxx"]) / 2, (t["miny"] + t["maxy"]) / 2
    return t


def describe(panel: Path, frequency: str, tag: str) -> pd.DataFrame:
    """Missing share, mean, sd, min, median, max of every variable of one period, one
    column at a time. For a true/false variable the mean is the share that is true."""
    out = {}
    for layer in LAYERS:
        path = layer_path(panel, frequency, layer, tag)
        for name in pq.read_schema(path).names:
            if name == "chip_id":
                continue
            a = pq.read_table(path, columns=[name]).column(name)
            n = len(a)
            if pa.types.is_boolean(a.type):
                missing = a.null_count
                true = pc.sum(a).as_py() or 0
                out[name] = {"missing": missing / n, "mean": true / (n - missing) if n > missing else np.nan}
            else:
                v = np.asarray(a.to_numpy(zero_copy_only=False), dtype=np.float64)
                ok = v[np.isfinite(v)]
                out[name] = {"missing": 1 - len(ok) / n} | ({"mean": ok.mean(), "sd": ok.std(), "min": ok.min(),
                                                              "median": np.median(ok), "max": ok.max()} if len(ok) else {})
    return pd.DataFrame(out).T.reindex(columns=["missing", "mean", "sd", "min", "median", "max"]).rename_axis("variable")
