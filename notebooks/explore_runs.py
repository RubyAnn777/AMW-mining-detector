import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Run checks")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "panel_pipeline"))          # status.py
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    import status
    from config import load_config
    from notebook_kit import cached, can_write, find_runs, is_cached
    from periods import Period
    from read_panel import panel_dir, read_panel
    from variables import dictionary

    _ = alt.data_transformers.disable_max_rows()
    VOTES = [f"vote_{i}" for i in range(1, 7)]
    CHIPS_PER_TILE = 529                  # 23 x 23 chips of 48 px, moved 24 px at a time
    return (CHIPS_PER_TILE, Path, Period, VOTES, alt, cached, can_write, dictionary, ds, find_runs, is_cached,
            json, load_config, mo, np, os, pa, panel_dir, pd, plt, pq, read_panel, status)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run checks

    The pipeline's own quality control for one run of `plan.py`, `detect.py` and
    `assemble_period.py`. It cross-checks `detect/` against the plan, against the assembled
    panel and against `compare.py`'s results:

    1. the run's settings and every task's state,
    2. the chip grid: every `chip_id` taken apart again and checked against the tile keys,
    3. every task, streamed: integrity checks and statistics per period and subregion,
    4. the border tiles, scored under both subregions: do the two copies agree, and did the
       panel keep the right one,
    5. the assembled panel against `detect/`, value by value on a sample of tiles,
    6. months against their quarter on a sample of tiles,
    7. detections and cloud cover through time and on the map,
    8. our quarters against AMW's published detections,
    9. what the full run weighs on disk.

    The panel itself, for reading and analysis, is in `panel_overview`. This notebook only
    reads; half-written files start with `_` and are skipped, so it is safe while jobs are
    writing. Sections 2 to 6 read a lot of data; their results are stored in
    `<run>/summaries/` and recomputed only when the run changes.
    """)
    return


@app.cell(hide_code=True)
def _(find_runs, mo, os):
    _default = mo.cli_args().get("run") or os.environ.get("RUN") or ""
    _runs = find_runs(_default)
    _name = os.path.basename(str(_default).rstrip("/"))
    run_picker = (mo.ui.dropdown(options=_runs, value=_name if _name in _runs else None, label="Run", full_width=True)
                  if _runs else mo.ui.text(value=str(_default), label="Run folder", full_width=True))
    run_picker
    return (run_picker,)


@app.cell(hide_code=True)
def _(Path, json, load_config, mo, pd, run_picker, status):
    RUN = Path(run_picker.value or ".")
    mo.stop(not (RUN / "run" / "config.yaml").is_file(),
            mo.callout(mo.md(f"`{RUN}` is not a planned run (no `run/config.yaml`)."), kind="warn"))
    cfg = load_config(RUN / "run" / "config.yaml")             # the frozen copy every script reads
    prov = json.loads((RUN / "run" / "provenance.json").read_text())
    tiles = pd.read_csv(RUN / "plan" / "tiles.csv")
    tiles_planned = status.tiles_per_subregion(RUN)
    planned_subregions = sorted(tiles_planned)
    border_keys = set(tiles["tile_key"][tiles["tile_key"].duplicated()])   # listed under two subregions

    def tile_number(key: str) -> int:
        """The part of a chip_id before its last three digits (grid.py), from a tile key 552:12:10.0:zone:col:row."""
        zone, col, row = (int(x) for x in key.split(":")[3:6])
        return (zone * 1000 + col + 500) * 10_000 + row + 5000

    tiles["tile"] = tiles["tile_key"].map(tile_number)
    tiles_of = {n: set(g["tile"]) for n, g in tiles.groupby("subregion")}

    def task_folder(frequency: str, period: str, subregion: int) -> Path:
        return RUN / "detect" / frequency / f"period={period}" / f"subregion={subregion}"

    mo.md(f"Planned {prov['planned_utc']} on {prov['host']}, commit `{str(prov['git_commit'])[:7]}`: "
          f"{tiles['tile_key'].nunique():,} tiles in subregions {planned_subregions}, {len(border_keys)} of them on borders.")
    return RUN, border_keys, cfg, planned_subregions, prov, task_folder, tile_number, tiles, tiles_of, tiles_planned


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The run: settings, plan and tasks

    One run is one folder, made once by `plan.py`: `run/` (the frozen `config.yaml` and
    `provenance.json`), `plan/` (`tiles.csv`, which tiles each subregion scores, and
    `chips.parquet`, every chip with its square and flags), `detect/<frequency>/period=<tag>/subregion=<n>/`
    (one task each: part files, `_status.json`, `_log.txt`) and, after assembly, the panel
    folder. The grid is `status.py`'s.
    """)
    return


@app.cell(hide_code=True)
def _(RUN, cfg, json, mo, pd, planned_subregions, prov, status, task_folder, tiles_planned):
    _rows = status.tasks(cfg, RUN, list(cfg.periods), tiles_planned)
    _grids = []
    for _freq in cfg.periods:
        _mine = [r for r in _rows if r["frequency"] == _freq]
        _states = pd.Series([r["state"] for r in _mine]).value_counts()
        _grids.append(f"{_freq}: " + ", ".join(f"{v} {k}" for k, v in _states.items()))
        _grids += status.grid(_mine, _freq, planned_subregions) + [""]
    _records = []
    for _r in _rows:
        _path = task_folder(_r["frequency"], _r["period"].tag, _r["subregion"]) / "_status.json"
        if not _path.is_file():
            continue
        _s = json.loads(_path.read_text())
        _ends = [(x.get("finished_utc") or x.get("stopped_utc"), x["started_utc"]) for x in _s["sessions"]]
        _hours = sum((pd.Timestamp(e) - pd.Timestamp(b)).total_seconds() for e, b in _ends if e) / 3600
        _last = _s["sessions"][-1]
        _records.append({"frequency": _r["frequency"], "period": _r["period"].tag, "subregion": _r["subregion"],
                         "state": _r["state"], "complete": _s["complete"], "tiles_saved": _s["tiles_saved"],
                         "tiles_planned": _s["tiles_planned"], "tiles_failed": len(_s["tiles_failed"]),
                         "chips_saved": _s["chips_saved"], "sessions": len(_s["sessions"]), "hours": round(_hours, 2),
                         "workers": _last.get("max_workers"), "limit": _last.get("limit"),
                         "ended": _last.get("finished_utc") or _last.get("stopped_utc"),
                         "git_commit": str(_last.get("git_commit"))[:7],
                         "image_cache": any(x.get("image_cache") for x in _s["sessions"])})
    started = pd.DataFrame(_records, columns=["frequency", "period", "subregion", "state", "complete", "tiles_saved",
                                              "tiles_planned", "tiles_failed", "chips_saved", "sessions", "hours",
                                              "workers", "limit", "ended", "git_commit", "image_cache"])
    STATUS_FILES = [task_folder(r.frequency, r.period, r.subregion) / "_status.json" for r in started.itertuples()]
    _settings = pd.DataFrame({"setting": ["periods", "region", "imagery", "inference", "thresholds", "counts"],
                              "value": [str({f: f"{len(p)} ({p[0].tag} to {p[-1].tag})" for f, p in cfg.periods.items()}),
                                        str(cfg.region.clip_to or "whole Amazon_ACA"), str(cfg.imagery),
                                        str(cfg.inference), str(cfg.thresholds), str(prov["counts"])]})
    mo.vstack([
        mo.ui.table(_settings, selection=None, pagination=False, wrapped_columns=["value"]),
        mo.callout(mo.md(f"Planned with uncommitted files: {prov['git_uncommitted_files']}"), kind="warn")
        if prov.get("git_uncommitted_files") else mo.md(""),
        mo.accordion({"Task grid": mo.plain_text("\n".join(_grids) + status.LEGEND)}, multiple=True),
        mo.ui.table(started, selection=None, page_size=12) if len(started) else mo.md("No task has written a `_status.json` yet."),
    ])
    return STATUS_FILES, started


@app.cell(hide_code=True)
def _(mo, planned_subregions, started):
    _with_tasks = sorted(set(started["subregion"])) if len(started) else []
    _quarters = sorted(started[(started["frequency"] == "quarterly") & started["complete"]]["period"].unique()) if len(started) else []
    _arg = mo.cli_args().get("quarter")
    subregion_picker = mo.ui.multiselect(options=[str(n) for n in planned_subregions],
                                         value=[str(n) for n in _with_tasks], label="Subregions")
    focus_picker = mo.ui.dropdown(options=_quarters, value=_arg if _arg in _quarters else (_quarters[0] if _quarters else None),
                                  label="Focus quarter (sections 6 to 8)")
    sample_picker = mo.ui.dropdown(options={"500 tiles": 500, "2,000 tiles": 2000, "every tile": 0}, value="2,000 tiles",
                                   label="Sample per subregion (section 6)")
    mo.hstack([subregion_picker, focus_picker, sample_picker], justify="start", gap=2)
    return focus_picker, sample_picker, subregion_picker


@app.cell(hide_code=True)
def _(focus_picker):
    focus = focus_picker.value
    return (focus,)


@app.cell(hide_code=True)
def _(RUN, can_write, mo, panel_dir, subregion_picker):
    subregions = sorted(int(n) for n in subregion_picker.value)
    PANEL = panel_dir(RUN, subregions)
    compute_now = mo.ui.run_button(label="Compute missing summaries now", kind="warn")
    mo.vstack([
        mo.md(f"Looking at subregions {subregions}; their panel folder: `{PANEL.name}/` "
              + ("(exists)" if PANEL.is_dir() else "(nothing assembled yet)")),
        mo.md("") if can_write(RUN) else mo.hstack([mo.md("<small>The run is read-only for you, so sections whose "
                                                          "summaries are not stored yet wait for this button:</small>"),
                                                    compute_now], justify="start"),
    ])
    return PANEL, compute_now, subregions


@app.cell(hide_code=True)
def _(RUN, can_write, compute_now, is_cached):
    def ready(name, inputs):
        """True if a section may show: its summary is stored, or this user may compute it."""
        return is_cached(RUN, name, inputs) or can_write(RUN) or compute_now.value
    return (ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The chip grid and the chip ID

    A tile is AMW's 552 px square (plus 12 px padding, 5.76 km) on a fixed global grid. The
    model scores chips of 48 px, moved 24 px at a time: 23 × 23 = 529 per tile. `chip_id` is
    one number built from where the chip is, so it is the same in every period, frequency
    and run:

    ```
    chip_id = zone | tile column + 500 | tile row + 5000 | chip index
              20   |      510          |     5103        |   000      ->  205105103000
    ```

    This reads `plan/chips.parquet` in batches and takes every `chip_id` apart again,
    against the tile key AMW's own code wrote. If it passes, the IDs are a safe join key
    between periods, frequencies, runs and the panel, and the chips of a tile are exactly
    `tile × 1000 + 0 … 528`, which the sections below rely on.
    """)
    return


@app.cell(hide_code=True)
def _(CHIPS_PER_TILE, RUN, cached, mo, np, pd, pq, ready, tiles):
    def _decode(ids):
        rest, chip = np.divmod(ids, 1000)
        rest, row = np.divmod(rest, 10_000)
        zone, col = np.divmod(rest, 1000)
        return zone, col - 500, row - 5000, chip

    def _grid():
        centre_chip = 11 * 23 + 11                   # the middle chip of the 23 x 23 grid
        pf = pq.ParquetFile(RUN / "plan" / "chips.parquet")
        ok = {"chip_id strictly increasing": True, "zone = tile key": True, "column = tile key": True,
              "row = tile key": True, "chip = chip_index": True}
        last, per_tile, centres = -1, {}, []
        for batch in pf.iter_batches(batch_size=2_000_000, columns=["chip_id", "tile_key", "chip_index", "minx", "miny", "maxx", "maxy"]):
            ids = batch.column("chip_id").to_numpy()
            zone, col, row, chip = _decode(ids)
            parts = batch.column("tile_key").to_pandas().str.split(":", expand=True)[[3, 4, 5]].astype(int).to_numpy()
            ok["chip_id strictly increasing"] &= bool(ids[0] > last and (np.diff(ids) > 0).all())
            ok["zone = tile key"] &= bool((zone == parts[:, 0]).all())
            ok["column = tile key"] &= bool((col == parts[:, 1]).all())
            ok["row = tile key"] &= bool((row == parts[:, 2]).all())
            ok["chip = chip_index"] &= bool((chip == batch.column("chip_index").to_numpy()).all())
            last = ids[-1]
            for t, c in zip(*np.unique(ids // 1000, return_counts=True)):
                per_tile[t] = per_tile.get(t, 0) + int(c)
            mid = chip == centre_chip
            centres.append(pd.DataFrame({"tile": ids[mid] // 1000,
                                         "x": (batch.column("minx").to_numpy()[mid] + batch.column("maxx").to_numpy()[mid]) / 2,
                                         "y": (batch.column("miny").to_numpy()[mid] + batch.column("maxy").to_numpy()[mid]) / 2}))
        ok[f"{CHIPS_PER_TILE} chips per tile"] = all(c == CHIPS_PER_TILE for c in per_tile.values())
        ok["every planned tile has chips"] = set(per_tile) == set(tiles["tile"])
        first = pf.read_row_group(0).slice(0, 3).to_pandas()
        return {"ok": ok, "chips": pf.metadata.num_rows, "tiles": len(per_tile),
                "centres": pd.concat(centres, ignore_index=True), "first": first}

    _inputs = [RUN / "plan" / "chips.parquet"]
    mo.stop(not ready("grid", _inputs), mo.md("*Not computed yet.*"))
    with mo.status.spinner(title="Reading every chip of the plan..."):
        grid = cached(RUN, "grid", _grid, _inputs)
    centres = grid["centres"]
    _checks = pd.DataFrame({"check": list(grid["ok"]), "result": ["PASS" if v else "FAIL" for v in grid["ok"].values()]})
    mo.vstack([mo.md(f"{grid['chips']:,} chips in {grid['tiles']:,} tiles."),
               mo.ui.table(_checks, selection=None, pagination=False),
               mo.ui.table(grid["first"], selection=None, pagination=False, label="The plan's first rows")])
    return centres, grid


@app.cell(hide_code=True)
def _(CHIPS_PER_TILE, np, tiles_of):
    def chips_of_tiles(tile_numbers) -> np.ndarray:
        """Every chip_id of these tiles, sorted (valid because each tile has chips 0 to 528, section 2)."""
        t = np.sort(np.fromiter(tile_numbers, dtype=np.int64))
        return (t[:, None] * 1000 + np.arange(CHIPS_PER_TILE)).ravel()

    def planned_ids(n: int) -> np.ndarray:
        return chips_of_tiles(tiles_of[n])
    return chips_of_tiles, planned_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Every task: integrity checks and statistics

    One row per task (frequency, period, subregion). The part files are read in batches, so
    a 28-million-chip task costs a few hundred MB. The checks, each PASS or FAIL:

    | check | meaning |
    |:--|:--|
    | all planned chips present | a complete task holds exactly its subregion's planned chips (unfinished tasks are not checked) |
    | no unplanned chips | nothing outside the plan |
    | no duplicate chip | within one task; border chips *are* duplicated across subregions (section 4) |
    | confidence = mean(votes) | AMW's score is the mean of the six members |
    | 0 ≤ confidence ≤ 1, n_clear ≤ n_obs | sanity |
    | no missing scores or counts | `confidence`, the votes, `n_obs`, `n_clear`, `empty_share` are never missing or infinite |

    Tasks scored between 22 Oct and 1 Nov 2026 (all of `aca_full_2026-10-23`) carry a
    `detect.py` bug: a pixel with no clear image arrived as `-inf` and was taken for a seen
    pixel. In every chip with such a pixel, `cs_clear_mean` and `cirrus_mean` are `-inf`,
    `ndvi_mean` and `bsi_mean` are missing and `empty_share` stayed 0. The scores, votes and
    counts are right. `masked_chips` is the share of such chips; `assemble_period.py` turns
    the wrong values into missing ones.

    `raw_share` is the share above 0.4, `loose_main` the share at or above 0.43 (the first
    half of AMW's loose rule), `vote_sd` how much the six members disagree, `low_obs` the
    share with fewer than one clear image per pixel.
    """)
    return


@app.cell(hide_code=True)
def _(RUN, STATUS_FILES, VOTES, cached, cfg, ds, mo, np, pd, planned_ids, ready, started, task_folder):
    QUALITY = ["n_obs", "n_clear", "n_clear_median", "empty_share", "thin_share", "clear_day_mean",
               "cs_mean", "cs_clear_mean", "ndvi_mean", "bsi_mean", "cirrus_mean"]
    _CORE = ["confidence", *VOTES, "n_obs", "n_clear", "empty_share"]
    RAW = 0.4

    def _stream(folder):
        d = ds.dataset(folder, format="parquet")              # files starting with _ are skipped
        cols = [c for c in dict.fromkeys(["chip_id", *_CORE, *QUALITY]) if c in d.schema.names]
        acc = {"rows": 0, "conf_sum": 0.0, "raw": 0, "loose_main": 0, "n_clear_sum": 0.0, "n_obs_sum": 0.0,
               "empty_sum": 0.0, "low_obs": 0, "vote_sd_sum": 0.0, "core_bad": 0, "masked": 0, "quality_missing_in": set(),
               "clear_gt_obs": 0, "conf_min": 1.0, "conf_max": 0.0, "mean_ok": True}
        ids, per_tile = [], []
        for batch in d.to_batches(columns=cols, batch_size=1_000_000):
            b = batch.to_pandas()
            conf = b["confidence"].to_numpy()
            acc["core_bad"] += int((~np.isfinite(b[[c for c in _CORE if c in b]].to_numpy(dtype=np.float64))).sum())
            if "cs_clear_mean" in b:
                acc["masked"] += int(np.isneginf(b["cs_clear_mean"].to_numpy(dtype=np.float64)).sum())
            other = b[[c for c in QUALITY if c in b and c not in _CORE]]
            if len(other.columns):
                acc["quality_missing_in"] |= set(other.columns[(~np.isfinite(other.to_numpy(dtype=np.float64))).any(axis=0)])
            votes = b[VOTES].to_numpy(dtype=np.float64)
            acc["rows"] += len(b)
            acc["conf_sum"] += float(np.nansum(conf))
            acc["raw"] += int((conf > RAW).sum())
            acc["loose_main"] += int((conf >= cfg.thresholds.loose.t_main).sum())
            acc["n_clear_sum"] += float(b["n_clear"].sum())
            acc["n_obs_sum"] += float(b["n_obs"].sum())
            acc["empty_sum"] += float(b["empty_share"].sum())
            acc["low_obs"] += int((b["n_clear"] < 1).sum())
            acc["vote_sd_sum"] += float(votes.std(axis=1).sum())
            acc["mean_ok"] &= bool(np.allclose(conf, votes.mean(axis=1), atol=1e-6))
            acc["clear_gt_obs"] += int((b["n_clear"] > b["n_obs"] + 1e-4).sum())
            acc["conf_min"] = min(acc["conf_min"], float(np.nanmin(conf)))
            acc["conf_max"] = max(acc["conf_max"], float(np.nanmax(conf)))
            ids.append(b["chip_id"].to_numpy())
            per_tile.append(b.assign(tile=b["chip_id"] // 1000, raw=conf > RAW)
                            .groupby("tile").agg(conf_max=("confidence", "max"), raw=("raw", "sum"),
                                                 n_clear=("n_clear", "sum"), chips=("chip_id", "size")))
        ids = np.sort(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)   # part files are not sorted
        per_tile = (pd.concat(per_tile).groupby(level=0).agg(conf_max=("conf_max", "max"), raw=("raw", "sum"),
                                                              n_clear=("n_clear", "sum"), chips=("chips", "sum"))
                    if per_tile else pd.DataFrame(columns=["conf_max", "raw", "n_clear", "chips"]))
        per_tile["n_clear"] = per_tile["n_clear"] / per_tile["chips"]
        return acc, per_tile, ids

    def _all_tasks():
        rows, tile_rows = [], []
        for t in started.itertuples():
            folder = task_folder(t.frequency, t.period, t.subregion)
            if not list(folder.glob("part_*.parquet")):
                continue
            acc, per_tile, ids = _stream(folder)
            want, n = planned_ids(t.subregion), acc["rows"]
            rows.append({"frequency": t.frequency, "period": t.period, "subregion": t.subregion, "complete": t.complete,
                         "rows": n, "chips_saved": t.chips_saved, "rows = chips_saved": n == t.chips_saved,
                         "all planned chips present": bool(np.array_equal(ids, want)) if t.complete else None,
                         "no unplanned chips": bool(np.isin(ids, want).all()),
                         "no duplicate chip": bool((np.diff(ids) > 0).all()),
                         "confidence = mean(votes)": acc["mean_ok"],
                         "0 <= confidence <= 1": 0 <= acc["conf_min"] and acc["conf_max"] <= 1,
                         "n_clear <= n_obs": acc["clear_gt_obs"] == 0,
                         "no missing scores or counts": acc["core_bad"] == 0,
                         "conf_mean": acc["conf_sum"] / n, "raw_count": acc["raw"], "raw_share": acc["raw"] / n,
                         "loose_main": acc["loose_main"] / n, "vote_sd": acc["vote_sd_sum"] / n,
                         "n_obs": acc["n_obs_sum"] / n, "n_clear": acc["n_clear_sum"] / n,
                         "empty_share": acc["empty_sum"] / n, "low_obs": acc["low_obs"] / n,
                         "masked_chips": acc["masked"] / n, "quality_missing_in": ", ".join(sorted(acc["quality_missing_in"]))})
            tile_rows.append(per_tile.reset_index().assign(frequency=t.frequency, period=t.period, subregion=t.subregion))
        return pd.DataFrame(rows), (pd.concat(tile_rows, ignore_index=True) if tile_rows else pd.DataFrame())

    mo.stop(not ready("tasks", STATUS_FILES), mo.md("*Not computed yet.*"))
    with mo.status.spinner(title="Streaming every task's part files..."):
        all_stats, all_tile_stats = cached(RUN, "tasks", _all_tasks, STATUS_FILES)
    return QUALITY, RAW, all_stats, all_tile_stats


@app.cell(hide_code=True)
def _(Period, all_stats, all_tile_stats, mo, subregions):
    stats = all_stats[all_stats["subregion"].isin(subregions)].copy() if len(all_stats) else all_stats
    tile_stats = all_tile_stats[all_tile_stats["subregion"].isin(subregions)] if len(all_tile_stats) else all_tile_stats
    mo.stop(not len(stats), mo.md("No task of these subregions has part files yet."))
    stats["period_start"] = stats["period"].map(lambda t: Period.parse(t).start)
    _checks = [c for c in stats.columns if c.startswith(("all ", "no ", "rows =", "confidence =", "0 <=", "n_clear <="))]
    _failed = int(stats[_checks].eq(False).any(axis=1).sum())
    mo.vstack([
        mo.callout(mo.md("All task checks pass." if not _failed else f"**{_failed} task(s) fail a check**: look for false below."),
                   kind="success" if not _failed else "danger"),
        mo.ui.tabs({
            "checks": mo.ui.table(stats[["frequency", "period", "subregion", "complete", "rows", *_checks]], selection=None, page_size=12),
            "statistics": mo.ui.table(stats[["frequency", "period", "subregion", "rows", "conf_mean", "raw_count", "raw_share",
                                             "loose_main", "vote_sd", "n_obs", "n_clear", "empty_share", "low_obs",
                                             "masked_chips", "quality_missing_in"]].round(5), selection=None, page_size=12),
        }),
    ])
    return stats, tile_stats


@app.cell(hide_code=True)
def _(dictionary, mo, pq, started, task_folder):
    _done = started[started["complete"]].sort_values(["frequency", "period", "subregion"]) if len(started) else started
    mo.stop(not len(_done), mo.md("No complete task yet."))
    _t = _done.iloc[0]
    _first = sorted(task_folder(_t["frequency"], _t["period"], _t["subregion"]).glob("part_*.parquet"))[0]
    _sample = pq.ParquetFile(_first).read_row_group(0).to_pandas().head(5)
    mo.accordion({f"The columns detect.py wrote ({_t['frequency']} {_t['period']}, subregion {_t['subregion']})": mo.vstack([
        mo.ui.table(_sample, selection=None, pagination=False),
        mo.ui.table(dictionary(list(_sample.columns)), selection=None, pagination=False, wrapped_columns=["meaning", "missing when"]),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Border tiles: scored twice, kept once

    A tile on the border between two subregions is listed under both, so `detect.py` scores
    it twice, in two tasks, from the same Earth Engine request. The two copies should be
    identical, or nearly: Earth Engine's data for recent months can change between the two
    scoring dates. The panel keeps one row per chip by AMW's rule (`postprocess.py:146`): the
    copy with the higher confidence, on a tie the lower subregion.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RAW, RUN, STATUS_FILES, border_keys, cached, chips_of_tiles, ds, json, mo, np, pa, pd, ready,
      started, subregions, task_folder, tile_number, tiles_of):
    pairs = [(n, m) for i, n in enumerate(subregions) for m in subregions[i + 1:] if tiles_of[n] & tiles_of[m]]

    def _border():
        rows, dropped = [], {}
        done_tasks = started[started["complete"] & started["subregion"].isin(subregions)]
        for (freq, period), g in done_tasks.groupby(["frequency", "period"]):
            for n, m in pairs:
                if not {n, m} <= set(g["subregion"]):
                    continue
                shared = chips_of_tiles(tiles_of[n] & tiles_of[m])
                copies = [ds.dataset(task_folder(freq, period, sub), format="parquet")
                          .to_table(columns=["chip_id", "confidence", "n_obs", "n_clear"], filter=ds.field("chip_id").isin(pa.array(shared)))
                          .to_pandas() for sub in (n, m)]
                both = copies[0].merge(copies[1], on="chip_id", suffixes=(f"_{n}", f"_{m}"), validate="one_to_one")
                cn, cm = both[f"confidence_{n}"], both[f"confidence_{m}"]
                gap = (cn - cm).abs()
                dropped[(freq, period)] = dropped.get((freq, period), 0) + int((np.minimum(cn, cm) > RAW).sum())
                row = {"frequency": freq, "period": period, "subregions": f"{n} & {m}", "chips": len(both),
                       "both copies present": len(both) == len(shared), "identical scores": float((gap == 0).mean()),
                       "max |score gap|": float(gap.max()), "flips across 0.4": int(((cn > RAW) != (cm > RAW)).sum()),
                       "identical n_obs": float((both[f"n_obs_{n}"] == both[f"n_obs_{m}"]).mean())}
                layer = PANEL / freq / "scores" / f"period={period}"
                if (layer / "_record.json").is_file():
                    rec = json.loads((layer / "_record.json").read_text()).get("border", {})
                    kept = pd.read_parquet(layer / "data.parquet", columns=["chip_id", "subregion", "confidence"],
                                           filters=[("chip_id", "in", shared)])
                    k = both.merge(kept, on="chip_id", validate="one_to_one")
                    tie = k[f"confidence_{n}"] == k[f"confidence_{m}"]
                    row |= {"panel kept the higher": bool((k["confidence"] == np.maximum(k[f"confidence_{n}"], k[f"confidence_{m}"])).all()),
                            "ties from the lower subregion": bool((k.loc[tie, "subregion"] == min(n, m)).all()),
                            "panel kept from": str(dict(k["subregion"].value_counts().sort_index())),
                            "record: copies differ": rec.get("copies_differ"), "record: largest gap": rec.get("largest_confidence_gap")}
                rows.append(row)
        return pd.DataFrame(rows), dropped

    _inputs = STATUS_FILES + sorted(PANEL.glob("*/scores/period=*/_record.json"))
    _name = f"border_{PANEL.name}"
    mo.stop(not ready(_name, _inputs), mo.md("*Not computed yet.*"))
    with mo.status.spinner(title="Reading the border chips of every task..."):
        border, dropped_raw = cached(RUN, _name, _border, _inputs)
    mo.vstack([
        mo.md(f"{len(border_keys):,} border tiles in the plan; between the subregions looked at: "
              + (", ".join(f"{n} & {m}: {len(tiles_of[n] & tiles_of[m]) * 529:,} chips" for n, m in pairs) or "none")),
        mo.ui.table(border, selection=None, page_size=12) if len(border) else
        mo.md("No period is complete for two subregions that share tiles."),
    ])
    return (dropped_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The assembled panel against `detect/`

    `check_panel.py` checks the panel's own consistency; this checks it against `detect/`, per
    period:

    - the rows are exactly the plan's chips of these subregions, once each, in order;
    - `det_raw` is `confidence > 0.4` wherever `scored`, and the flags are missing exactly where not;
    - the panel's raw detections equal detect's once the border copies are resolved;
    - on 200 random tiles per subregion, every value of the `scores` layer equals detect's kept
      copy exactly, except the five columns the assembly corrects for the masked-pixel bug
      (`empty_share`, `cs_clear_mean`, `ndvi_mean`, `bsi_mean`, `cirrus_mean`), which are
      compared after applying the same rule (`mark_lost_quality` in `assemble_period.py`).

    Not a difference: 368 chip pairs at the 60°W zone seam have two `chip_id`s but one AMW
    centre key. Both rows stay in the panel; the lower-scored one gets no loose or stringent
    flag (`twins_dropped` in the postprocess record).
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RAW, RUN, STATUS_FILES, VOTES, cached, cfg, chips_of_tiles, dropped_raw, ds, mo, np, pa, pd,
      planned_ids, read_panel, ready, stats, subregions, task_folder, tiles_of):
    _MASKED = ["empty_share", "cs_clear_mean", "ndvi_mean", "bsi_mean", "cirrus_mean"]
    _EXACT = ["confidence", *VOTES, "n_obs", "n_clear", "n_clear_median", "thin_share", "clear_day_mean", "cs_mean"]
    _EQUALITY_TILES = 200

    def expected_raw(freq, period):
        """detect's raw detections for the period, border chips counted once (the kept copy)."""
        mine = stats[(stats["frequency"] == freq) & (stats["period"] == period)] if len(stats) else pd.DataFrame()
        if not len(mine) or not mine["complete"].all() or set(mine["subregion"]) != set(subregions):
            return None
        return int(mine["raw_count"].sum()) - dropped_raw.get((freq, period), 0)

    def _lost_quality(d):
        d = d.copy()
        d.loc[np.isneginf(d["cs_clear_mean"].to_numpy(dtype=np.float64)), "empty_share"] = np.nan
        d.loc[d["n_clear"] == 0, "empty_share"] = 1.0
        for c in _MASKED[1:]:
            v = d[c].to_numpy(dtype=np.float64)
            d[c] = np.where(np.isfinite(v), v, np.nan)
        return d

    def _compare():
        rng = np.random.default_rng(1)
        sample_tiles = [t for n in subregions for t in rng.choice(sorted(tiles_of[n]), min(_EQUALITY_TILES, len(tiles_of[n])), replace=False)]
        equality_ids = chips_of_tiles(sample_tiles)
        planned_here = np.unique(np.concatenate([planned_ids(n) for n in subregions]))

        def detect_rows(freq, period):
            parts = []
            for sub in subregions:
                d = ds.dataset(task_folder(freq, period, sub), format="parquet")
                cols = [c for c in ["chip_id", *_EXACT, *_MASKED] if c in d.schema.names]
                parts.append(d.to_table(columns=cols, filter=ds.field("chip_id").isin(pa.array(equality_ids))).to_pandas().assign(subregion=sub))
            d = pd.concat(parts).sort_values(["chip_id", "confidence", "subregion"], ascending=[True, False, True])
            return d.drop_duplicates("chip_id").set_index("chip_id")

        rows = []
        for freq in cfg.periods:
            base = PANEL / freq / "scores"
            for folder in sorted(base.glob("period=*")) if base.is_dir() else []:
                if not (folder / "_record.json").is_file():
                    continue
                period = folder.name.removeprefix("period=")
                p = read_panel(RUN, freq, periods=[period], subregions=subregions,
                               columns={"scores": ["subregion", "scored", "confidence"],
                                        "postprocess": ["det_raw", "det_andes", "det_loose", "det_stringent"]})
                scored = p["scored"].to_numpy()
                raw_from_conf = (p["confidence"].to_numpy() > RAW) & scored
                exp = expected_raw(freq, period)
                row = {"frequency": freq, "period": period, "rows": len(p),
                       "rows = planned chips": len(p) == len(planned_here),
                       "chip_id = plan, in order": bool(np.array_equal(p["chip_id"].to_numpy(), planned_here)),
                       "scored": float(scored.mean()),
                       "det_raw = confidence > 0.4": bool((p["det_raw"].fillna(False).to_numpy(dtype=bool) == raw_from_conf).all()),
                       "flags missing exactly where unscored": bool((p["det_raw"].isna().to_numpy() == ~scored).all()),
                       "det_raw": int(p["det_raw"].sum()), "detect's raw, borders resolved": exp,
                       "panel = detect": (int(p["det_raw"].sum()) == exp) if exp is not None else None,
                       "det_loose": int(p["det_loose"].sum()), "det_stringent": int(p["det_stringent"].sum()),
                       "det_andes": int(p["det_andes"].sum())}
                del p
                theirs = pd.read_parquet(folder / "data.parquet", filters=[("chip_id", "in", equality_ids)]).set_index("chip_id")
                theirs = theirs[theirs["scored"].to_numpy(dtype=bool)]
                mine = detect_rows(freq, period).reindex(theirs.index)
                differ = [c for c in _EXACT if c in mine and c in theirs
                          and not np.array_equal(mine[c].to_numpy(dtype=np.float64), theirs[c].to_numpy(dtype=np.float64), equal_nan=True)]
                masked = [c for c in _MASKED if c in mine and c in theirs]
                if masked:
                    fixed = _lost_quality(mine)
                    differ += [f"{c} (after the rule)" for c in masked
                               if not np.allclose(fixed[c].to_numpy(dtype=np.float64), theirs[c].to_numpy(dtype=np.float64), atol=1e-6, equal_nan=True)]
                if "subregion" in theirs and not np.array_equal(mine["subregion"].to_numpy(), theirs["subregion"].to_numpy()):
                    differ.append("subregion (the kept copy)")
                rows.append(row | {"sample chips": len(theirs), "values = detect": not differ, "differ in": ", ".join(differ)})
        return pd.DataFrame(rows)

    _inputs = STATUS_FILES + sorted(PANEL.glob("*/*/period=*/_record.json"))
    _name = f"panel_vs_detect_{PANEL.name}"
    mo.stop(not PANEL.is_dir(), mo.md(f"Nothing assembled yet under `{PANEL.name}/`."))
    mo.stop(not ready(_name, _inputs), mo.md("*Not computed yet.*"))
    with mo.status.spinner(title="Comparing the panel with detect's output..."):
        panel_check = cached(RUN, _name, _compare, _inputs)
    _flags = [c for c in panel_check.columns if panel_check[c].dtype == bool or c in ("panel = detect", "values = detect")]
    _bad = panel_check[_flags].eq(False).any().any() if len(panel_check) else False
    mo.vstack([
        mo.callout(mo.md("All panel checks pass." if not _bad else "**Some panel checks fail**: look for false below."),
                   kind="success" if not _bad else "danger"),
        mo.ui.table(panel_check, selection=None, page_size=12),
    ])
    return expected_raw, panel_check


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Months against their quarter, on a sample of tiles

    Both frequencies come from the same run, so they share `chip_id`. A quarter's window is
    exactly its three months' windows put together, so per pixel the quarter's image counts
    equal the sum of its months' counts, and the identity holds per chip (`findings.md` §1).
    The scores do not add up: a quarter's median image is its own composite. A mismatch in the
    counts would mean a date bug, or Earth Engine data that changed between the two scoring
    dates.
    """)
    return


@app.cell(hide_code=True)
def _(Period, RAW, RUN, STATUS_FILES, alt, cached, chips_of_tiles, ds, focus, json, mo, np, pa, pd, ready,
      sample_picker, subregions, task_folder, tiles_of):
    mo.stop(focus is None, mo.md("No complete quarter yet."))
    _months = [m.tag for m in Period.parse(focus).months()]

    def _complete(freq, period, sub):
        f = task_folder(freq, period, sub) / "_status.json"
        return f.is_file() and json.loads(f.read_text())["complete"]

    def _months_vs_quarter():
        frames, notes = [], []
        for sub in subregions:
            have = [p for p in _months if _complete("monthly", p, sub)]
            if not _complete("quarterly", focus, sub) or not have:
                notes.append(f"subregion {sub}: {focus} or its months not complete, skipped")
                continue
            pool = np.array(sorted(tiles_of[sub]))
            if sample_picker.value and sample_picker.value < len(pool):
                pool = np.random.default_rng(0).choice(pool, sample_picker.value, replace=False)
            ids = pa.array(chips_of_tiles(pool))

            def load(freq, period):
                d = ds.dataset(task_folder(freq, period, sub), format="parquet")
                return (d.to_table(columns=["chip_id", "confidence", "n_obs", "n_clear"], filter=ds.field("chip_id").isin(ids))
                        .to_pandas().replace([np.inf, -np.inf], np.nan))

            q = load("quarterly", focus).rename(columns={"confidence": "conf_q", "n_obs": "n_obs_q", "n_clear": "n_clear_q"})
            m = pd.concat([load("monthly", p).assign(month=p) for p in have])
            agg = (m.groupby("chip_id").agg(months=("month", "nunique"), n_obs_m=("n_obs", "sum"), n_clear_m=("n_clear", "sum"),
                                            n_obs_ok=("n_obs", "count"), n_clear_ok=("n_clear", "count"),
                                            conf_max_m=("confidence", "max"), conf_mean_m=("confidence", "mean")).reset_index())
            frames.append(agg.merge(q, on="chip_id", validate="one_to_one").assign(subregion=sub))
            notes.append(f"subregion {sub}: {len(pool):,} tiles, {len(q):,} chips, months: {', '.join(have)}")
        return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), notes

    _name = f"months_{focus}_{'_'.join(map(str, subregions))}_{sample_picker.value}"
    mo.stop(not ready(_name, STATUS_FILES), mo.md("*Not computed yet.*"))
    with mo.status.spinner(title="Reading the sample tiles..."):
        _full, _notes = cached(RUN, _name, _months_vs_quarter, STATUS_FILES)
    mo.stop(_full.empty, mo.md("<br>".join(_notes)))
    _three = _full[_full["months"] == len(_months)]
    _lines = []
    for _c in ("n_obs", "n_clear"):
        _ok = _three[f"{_c}_ok"] == len(_months)
        _diff = (_three[f"{_c}_m"] - _three[f"{_c}_q"]).abs()[_ok].dropna()
        _lines.append(f"`{_c}`, sum of the months against the quarter, over {len(_diff):,} chips: largest difference "
                      f"{_diff.max():.4f}; equal within 0.001 for {(_diff <= 1e-3).mean():.3%} ({int((_diff > 1e-3).sum()):,} chips differ).")
    _cross = pd.crosstab(_three["conf_q"] > RAW, _three["conf_max_m"] > RAW, rownames=["quarter > 0.4"], colnames=["any month > 0.4"])
    _plot = _three[["conf_mean_m", "conf_q", "n_clear_m", "n_clear_q"]].sample(min(len(_three), 10_000), random_state=0)
    _a = alt.Chart(_plot).mark_circle(size=6, opacity=0.3).encode(
        x=alt.X("conf_mean_m:Q", title="mean of the monthly scores"), y=alt.Y("conf_q:Q", title="quarterly score"))
    _b = alt.Chart(_plot).mark_circle(size=6, opacity=0.3).encode(
        x=alt.X("n_clear_m:Q", title="sum of the monthly n_clear"), y=alt.Y("n_clear_q:Q", title="quarterly n_clear"))
    _line = alt.Chart(pd.DataFrame({"v": [0, float(max(_plot["n_clear_q"].max(), _plot["n_clear_m"].max(), 1))]})).mark_line(
        color="grey", strokeWidth=0.8).encode(x="v:Q", y="v:Q")
    _unit = alt.Chart(pd.DataFrame({"v": [0, 1]})).mark_line(color="grey", strokeWidth=0.8).encode(x="v:Q", y="v:Q")
    mo.vstack([
        mo.md("<br>".join(_notes)), *[mo.md(l) for l in _lines],
        mo.md(f"{focus}: detected in the quarter (rows) against detected in any of its months (columns):"),
        mo.ui.table(_cross.rename(columns=str).reset_index().astype({"quarter > 0.4": str}), selection=None, pagination=False),
        mo.hstack([(_a + _unit).properties(width=380, height=320, title="Same chip: months against quarter"),
                   (_b + _line).properties(width=380, height=320, title="Count identity (on the line)")]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Through time and on the map

    Per subregion and period: the raw detection share, clear images per pixel, and the share
    of pixels never seen clear. A cloudy month shows as low `n_clear` and high `empty_share`,
    and its detections are the model reading a half-black image; check this before reading a
    jump as mining. The maps are per tile, for the focus quarter and its months.
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, stats):
    _measure = {"raw_share": "share of chips > 0.4", "n_clear": "clear images per pixel (mean)",
                "empty_share": "share of pixels never seen clear"}
    _long = stats.melt(id_vars=["frequency", "subregion", "period", "period_start"], value_vars=list(_measure),
                       var_name="measure", value_name="value").assign(series=lambda d: d["frequency"] + " s" + d["subregion"].astype(str))
    _long["measure"] = _long["measure"].map(_measure)
    mo.ui.altair_chart(
        alt.Chart(_long).mark_line(point=True).encode(
            x=alt.X("period_start:T", title=None), y=alt.Y("value:Q", title=None),
            color=alt.Color("series:N", title=None), strokeDash=alt.StrokeDash("frequency:N", legend=None),
            tooltip=["series", "period", alt.Tooltip("value:Q", format=".4f")])
        .properties(width=280, height=220).facet(column=alt.Column("measure:N", title=None))
        .resolve_scale(y="independent"),
        chart_selection=False, legend_selection=True)
    return


@app.cell(hide_code=True)
def _(Period, centres, focus, mo, plt, tile_stats):
    mo.stop(not len(tile_stats) or focus is None)
    _views = [("quarterly", focus)] + [("monthly", m.tag) for m in Period.parse(focus).months()]
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    _sc = None
    for _ax, (_freq, _per) in zip(_axes.flat, _views):
        _sel = tile_stats[(tile_stats["frequency"] == _freq) & (tile_stats["period"] == _per)]
        _sel = _sel.sort_values("conf_max").drop_duplicates("tile", keep="last").merge(centres, on="tile")   # border tiles: the higher copy
        if _sel.empty:
            _ax.set_title(f"{_freq} {_per}: not scored")
            continue
        _sc = _ax.scatter(_sel["x"], _sel["y"], c=_sel["conf_max"], s=3, cmap="magma", vmin=0, vmax=1)
        _ax.set(title=f"{_per}: {int(_sel['raw'].sum()):,} chips > 0.4 in {int((_sel['raw'] > 0).sum()):,} tiles", aspect="equal")
    if _sc is not None:
        _fig.colorbar(_sc, ax=_axes, label="strongest chip of the tile (confidence)", shrink=0.6)
    _q = tile_stats[(tile_stats["frequency"] == "quarterly") & (tile_stats["period"] == focus)].drop_duplicates("tile").merge(centres, on="tile")
    _fig2, _ax2 = plt.subplots(figsize=(7, 6))
    _sc2 = _ax2.scatter(_q["x"], _q["y"], c=_q["n_clear"], s=3, cmap="viridis")
    _ax2.set(title=f"{focus}: clear images per pixel, by tile", aspect="equal")
    _fig2.colorbar(_sc2, ax=_ax2, label="mean n_clear")
    mo.ui.tabs({"strongest chip per tile": _fig, "clear images per tile": _fig2})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Against AMW's published detections

    `amw_comparison/compare.py` joins our quarters with AMW's raw detections on AMW's own
    chip key and writes `<run>/amw_comparison/`. The `amw_comparison` notebook goes into it in
    depth; here its counts are cross-checked: the chips we detect in that file must be the
    panel's raw detections of section 5 (or detect's, borders resolved).
    """)
    return


@app.cell(hide_code=True)
def _(RAW, RUN, expected_raw, focus, mo, np, panel_check, pd):
    mo.stop(focus is None)
    _dir = RUN / "amw_comparison"
    mo.stop(not (_dir / "joins.csv").is_file(), mo.md(f"No `compare.py` results under `{_dir}`."))
    mo.stop(not (_dir / f"chips_{focus}.parquet").is_file(),
            mo.md(f"`compare.py` results exist, but not for {focus}: {sorted(p.name for p in _dir.glob('chips_*.parquet'))}"))
    _c = pd.read_parquet(_dir / f"chips_{focus}.parquet")
    _ours, _theirs = _c["confidence"] > RAW, _c["amw_confidence"] > RAW
    _c["outcome"] = np.select([_ours & _theirs, _ours & ~_theirs, ~_ours & _theirs], ["both", "only ours", "only AMW"], "a month only")
    _table = _c.groupby(["subregion", "outcome"]).size().unstack(fill_value=0)
    _total = int(_ours.sum())
    _ref = panel_check[(panel_check["frequency"] == "quarterly") & (panel_check["period"] == focus)] if len(panel_check) else pd.DataFrame()
    _expected = int(_ref["det_raw"].iloc[0]) if len(_ref) else expected_raw("quarterly", focus)
    _verdict = ("no complete reference to check against" if _expected is None else
                f"{'the panel' if len(_ref) else 'detect, borders resolved,'} has {_expected:,}: "
                + ("match." if _expected == _total else "MISMATCH (compare.py keeps one row per AMW key, so the 60°W seam twins count once there; see joins.csv)."))
    mo.vstack([
        mo.ui.table(pd.read_csv(_dir / "amw_files.csv"), selection=None, pagination=False, label="AMW files"),
        mo.ui.table(_table.reset_index(), selection=None, pagination=False, label=f"{focus}, per subregion"),
        mo.md(f"{focus}: `compare.py` counts {_total:,} of our chips above 0.4; {_verdict}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. What the full run weighs

    Bytes per row measured on this run's part files, scaled to Amazon_ACA (about 145 million
    chips, 105 months and 35 quarters). Small files compress a little worse than big ones, so
    this errs high.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RUN, mo, pq):
    _parts = list((RUN / "detect").rglob("part_*.parquet"))
    _rows = sum(pq.ParquetFile(p).metadata.num_rows for p in _parts)
    _size = sum(p.stat().st_size for p in _parts)
    mo.stop(not _rows, mo.md("No part files yet."))
    _bpr = _size / _rows
    _lines = [f"Measured: {_rows:,} rows in {len(_parts):,} part files, {_bpr:.1f} bytes per row on disk, {_size / 1e9:.2f} GB."]
    for _name, _n in (("monthly", 105), ("quarterly", 35)):
        _lines.append(f"{_name}: {145e6 * _n / 1e9:.1f} billion rows, about {145e6 * _n * _bpr / 1e9:,.0f} GB.")
    _layers = list(PANEL.rglob("data.parquet")) if PANEL.is_dir() else []
    if _layers:
        _lines.append(f"Panel so far (`{PANEL.name}/`): {len(_layers)} layer files, {sum(p.stat().st_size for p in _layers) / 1e9:.2f} GB.")
    mo.md("<br>".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the data later.** Never `pd.read_parquet` the whole `detect/` folder. Filter while
    reading, so only the needed files and columns leave the disk:

    ```python
    d = ds.dataset(RUN / "detect" / "monthly", format="parquet", partitioning="hive")
    t = d.to_table(columns=["chip_id", "confidence", "n_clear"],
                   filter=(ds.field("period") == "2025-01") & (ds.field("subregion") == 3))
    ```

    The assembled panel reads layer by layer with
    `read_panel(RUN, "monthly", periods=[...], columns={...})` (`panel_pipeline/lib/read_panel.py`);
    the same files open in DuckDB and in R with `arrow::open_dataset()`.
    """)
    return


if __name__ == "__main__":
    app.run()
