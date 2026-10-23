"""Read the panel: chosen layers of one frequency, side by side, as one table.

    from read_panel import read_panel
    m = read_panel("tests_output/test_region_2026-10-19", "monthly")
    q = read_panel("tests_output/test_region_2026-10-19", "quarterly")
    both = m.merge(q, left_on=["chip_id", "quarter"], right_on=["chip_id", "period"],
                   suffixes=("", "_q"))        # each month next to its own quarter

Monthly and quarterly stay two separate panels (open_design_decisions.Rmd, P3). They meet
only through the `quarter` column of the monthly panel, which names the quarter a month
belongs to in the quarterly panel's own tags (2026-02 -> 2026Q1).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from periods import Period


def read_panel(run: str | Path, frequency: str, layers: tuple[str, ...] = ("scores", "postprocess"),
               periods: list[str] | None = None, columns: dict[str, list[str]] | None = None
               ) -> pd.DataFrame:
    """One row per (chip, period), with the columns of every layer asked for.

    periods: only these period tags (default: all assembled). columns: per layer, only
    these columns (chip_id is always read). The layers of a period hold the same chips in
    the same order, which is checked here rather than trusted.
    """
    base = Path(run) / "panel" / frequency
    tags = periods or sorted(p.name.removeprefix("period=")
                             for p in (base / layers[0]).glob("period=*")
                             if (p / "_record.json").exists())
    frames = []
    for tag in tags:
        parts = []
        for layer in layers:
            path = base / layer / f"period={tag}" / "data.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{frequency} {tag} has no {layer} layer")
            wanted = None if columns is None or layer not in columns else ["chip_id", *columns[layer]]
            part = pd.read_parquet(path, columns=wanted)
            if parts and not part["chip_id"].equals(parts[0]["chip_id"]):
                raise ValueError(f"{frequency} {tag}: the {layer} layer's rows don't line up")
            parts.append(part if not parts else part.drop(columns="chip_id"))
        period = Period.parse(tag)
        table = pd.concat(parts, axis=1)
        table.insert(1, "period", tag)
        table.insert(2, "month_index", period.month_index)   # one clock for both panels
        if frequency == "monthly":
            table.insert(3, "quarter", period.quarter().tag)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)
