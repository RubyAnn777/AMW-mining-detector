"""Put one period's scores together across subregions, and add AMW's postprocess flags.

Run after detect.py has finished every subregion of a period:
    python panel_pipeline/assemble_period.py --run tests_output/<run> --frequency monthly
    ... --subregions 2 3              only these subregions (default: all of the plan)
    ... --from 2026-01 --to 2026-03   only these periods (default: all of the run's)
    ... --allow-failed                accept tiles that failed every try (their chips get
                                      empty scores, scored = False)

Writes two layers of the panel per period. Both hold every chip of the chosen subregions'
tiles, once, sorted by chip_id, so their rows line up one to one:
    <run>/<panel>/<frequency>/scores/period=<tag>/data.parquet        detect's values
    <run>/<panel>/<frequency>/postprocess/period=<tag>/data.parquet   AMW's flags, distances
and a _record.json next to each: what went in, counts, settings, which code ran.
<panel> is "panel" for the whole plan and "panel_subregions_2_3" for --subregions 2 3, so
a panel of part of the region never mixes with the panel of all of it (lib/read_panel.py).

Nothing written is ever changed (open_design_decisions.Rmd, P1 and P2). A layer that
exists is skipped; later steps add new layers next to these. Run the same command again
at any time: it assembles what is ready and missing, and says why the rest isn't.

Memory: one period at a time. Measured: 2 GB and 40 s for a period of 10.5 M chips. It
grows with the chips: about 8 GB for subregions 2 and 3, 25 GB for all of Amazon_ACA
(estimates).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shapely
import yaml
from pyproj import Geod
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))   # our helpers: config, periods, grid, ...

import grid  # noqa: F401  (puts AMW's core/ on the search path, see grid.py)
from config import Config, load_config
from lock import FolderLock, Locked
from periods import Period
from provenance import environment
from read_panel import panel_name

# AMW code. This must come after `grid`, which puts AMW's core/ on the search path.
import postprocess as amw

LAYERS = ("scores", "postprocess")
VOTES = [f"vote_{i}" for i in range(1, 7)]
QUALITY = ["n_obs", "n_clear", "n_clear_median", "empty_share", "thin_share",
           "clear_day_mean", "cs_mean", "cs_clear_mean", "ndvi_mean", "bsi_mean",
           "cirrus_mean"]
ROW_GROUP_ROWS = 200 * 529     # 200 tiles of 529 chips. build_panel.py will read an area
                               # (a chip_id range) at a time; Parquet skips the row groups
                               # outside it, because each records its smallest and largest id
BATCH_ROWS = 1_000_000         # detect's part files are read this many rows at a time
GEOD = Geod(ellps="WGS84")     # distances on the Earth's real shape (open decision D3)


class NotReady(Exception):
    """The period can't be assembled yet; the message says why."""


class Broken(Exception):
    """The period's detect output contradicts itself or the plan; the message says how."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pick_periods(cfg: Config, frequency: str, first: str | None,
                 last: str | None) -> list[Period]:
    """The run's periods of one frequency, from `first` to `last` (None: all)."""
    # Same as detect.py:329, copied because importing detect.py would load TensorFlow.
    periods = cfg.periods[frequency]
    lo = Period.parse(first) if first else periods[0]
    hi = Period.parse(last) if last else periods[-1]
    if lo.frequency != frequency or hi.frequency != frequency:
        raise SystemExit(f"ASSEMBLE ERROR: --from and --to must be {frequency} periods, "
                         f"like {periods[0].tag}")
    chosen = [p for p in periods if lo <= p <= hi]
    if not chosen:
        raise SystemExit(f"ASSEMBLE ERROR: this run has no {frequency} periods "
                         f"from {lo.tag} to {hi.tag}")
    return chosen


def tile_number(key: str) -> int:
    """A tile key's number, the part of its chip_ids before the last three digits (grid.py)."""
    zone, ti, tj = (int(float(part)) for part in key.split(":")[3:])
    return (zone * 1000 + ti + 500) * 10_000 + tj + 5000


# ------------------------------------------------------------ the plan

class Plan:
    """The chips of the chosen subregions: their ids, in order, and what is fixed about them.

    Every period's layers have exactly these rows. Read once, used for every period.
    """

    def __init__(self, run_dir: Path, tiles: pd.DataFrame, subregions: list[int]):
        self.subregions = subregions
        self.chips_file = run_dir / "plan" / "chips.parquet"
        tiles = tiles[tiles["subregion"].isin(subregions)].assign(
            tile=lambda t: t["tile_key"].map(tile_number))
        # A border tile is listed once per subregion; its chips are one row, credited to
        # the lowest subregion until a score says otherwise (see place_scores).
        home = tiles.groupby("tile")["subregion"].min()
        self.tiles = home.index.to_numpy()                 # sorted, unique
        self.border_tiles = int((tiles.groupby("tile").size() > 1).sum())

        # The chip ids come from the plan's own file, not from a formula. Read in batches,
        # two columns, so the whole-region file (145 M chips) never sits in memory at once.
        ids, andes = [], []
        for batch in pq.ParquetFile(self.chips_file).iter_batches(
                batch_size=BATCH_ROWS, columns=["chip_id", "in_andes"]):
            chip_id = batch.column("chip_id").to_numpy()
            mine = np.isin(chip_id // 1000, self.tiles)
            ids.append(chip_id[mine])
            andes.append(batch.column("in_andes").to_numpy(zero_copy_only=False)[mine])
        self.ids = np.concatenate(ids)
        self.in_andes = np.concatenate(andes)
        if not (np.all(np.diff(self.ids) > 0)
                and np.array_equal(np.unique(self.ids // 1000), self.tiles)):
            raise SystemExit("ASSEMBLE ERROR: plan/chips.parquet and plan/tiles.csv disagree "
                             "(chips missing for a tile, or not sorted by chip_id)")
        tile_of_chip = np.searchsorted(self.tiles, self.ids // 1000)
        self.home = home.to_numpy().astype(np.int8)[tile_of_chip]   # subregion per chip

    def squares(self, ids: np.ndarray) -> np.ndarray:
        """The squares (minx, miny, maxx, maxy) of some chips, in the order of `ids`."""
        table = ds.dataset(self.chips_file).to_table(
            columns=["chip_id", "minx", "miny", "maxx", "maxy"],
            filter=pc.field("chip_id").isin(ids))
        found = table.column("chip_id").to_numpy()
        if len(found) != len(ids):
            raise Broken(f"{len(ids) - len(found)} scored chips have no square in the plan")
        order = np.argsort(found)
        where = order[np.searchsorted(found, ids, sorter=order)]
        return np.column_stack([table.column(c).to_numpy()[where]
                                for c in ("minx", "miny", "maxx", "maxy")])


# ------------------------------------------------------------ is the period ready?

def detect_inputs(run_dir: Path, frequency: str, period: Period,
                  subregions: list[int], allow_failed: bool) -> dict[int, dict]:
    """Each subregion's detect folder and its record, once every one is finished.

    Raises NotReady otherwise. Finished means: nobody is writing it, every tile was tried,
    and (unless allow_failed) none failed.
    """
    # ADDITION: AMW's concat assembles whatever subregion files it finds
    # (pipeline.py:328-357). We refuse a period with a subregion missing or half done,
    # because the isolation rule counts neighbours across subregion borders.
    found = {}
    for n in subregions:
        folder = (run_dir / "detect" / frequency / f"period={period.tag}"
                  / f"subregion={n}")
        if (folder / "_lock.json").exists():
            raise NotReady(f"subregion {n} is being scored, or its detect.py stopped "
                           f"without cleaning up (status.py shows which)")
        if not (folder / "_status.json").is_file():
            raise NotReady(f"subregion {n} is not scored yet")
        status = json.loads((folder / "_status.json").read_text())
        if status["tiles_not_tried"]:
            raise NotReady(f"subregion {n}: {status['tiles_saved']:,} of "
                           f"{status['tiles_planned']:,} tiles saved")
        if status["tiles_failed"] and not allow_failed:
            raise NotReady(f"subregion {n}: {len(status['tiles_failed'])} tile(s) failed; "
                           f"run its detect.py again to retry them, or use --allow-failed")
        found[n] = {"folder": folder, "status": status}
    members = {tuple(f["status"]["model_members"]) for f in found.values()}
    if len(members) > 1:
        raise Broken(f"the subregions were scored with different model members: {members}")
    return found


def inputs_summary(found: dict[int, dict]) -> dict:
    """What a period was assembled from, as written in its records (and compared later)."""
    return {str(n): {"chips_saved": f["status"]["chips_saved"],
                     "tiles_failed": sorted(f["status"]["tiles_failed"]),
                     "last_session_finished_utc": f["status"]["sessions"][-1]["finished_utc"]}
            for n, f in found.items()}


# ------------------------------------------------------------ layer 1: scores

def place_scores(found: dict[int, dict], plan: Plan) -> tuple[dict[str, np.ndarray], dict]:
    """Every chip's scores, in the plan's row order, from all subregions' part files.

    Each planned chip has a fixed row (its place in plan.ids). The part files are read a
    batch at a time and each value is put into its chip's row, so memory holds the
    finished layer and one batch, never all part files at once.
    """
    n = len(plan.ids)
    first = sorted(found[min(found)]["folder"].glob("part_*.parquet"))
    names = pq.read_schema(first[0]).names if first else []
    quality = [c for c in QUALITY if c in names]   # runs before 23 Oct have only three
    cols = {"subregion": np.zeros(n, dtype=np.int8),
            "scored": np.zeros(n, dtype=bool),
            "confidence": np.full(n, np.nan, dtype=np.float64),
            **{c: np.full(n, np.nan, dtype=np.float32) for c in VOTES + quality}}
    values = ["confidence", *VOTES, *quality]
    border = {"chips_scored_twice": 0, "copies_differ": 0, "largest_confidence_gap": 0.0}

    for sub in sorted(found):                       # lowest subregion first
        rows = 0
        seen = np.zeros(n, dtype=bool)              # the chips this subregion has given so far
        for part in sorted(found[sub]["folder"].glob("part_*.parquet")):
            for batch in pq.ParquetFile(part).iter_batches(batch_size=BATCH_ROWS,
                                                           columns=["chip_id", *values]):
                chip_id = batch.column("chip_id").to_numpy()
                row = np.searchsorted(plan.ids, chip_id)
                if (row >= n).any() or (plan.ids[np.minimum(row, n - 1)] != chip_id).any():
                    raise Broken(f"subregion {sub}'s part files hold chips that aren't "
                                 f"in the plan")
                if seen[row].any() or len(np.unique(row)) < len(row):
                    raise Broken(f"a chip is saved twice in subregion {sub}'s part files")
                seen[row] = True
                new = {c: batch.column(c).to_numpy(zero_copy_only=False) for c in values}   # null -> NaN

                # Open decision D7, now measured: a tile on a subregion border is scored
                # once per subregion, under the same chip_ids. Same as AMW: keep the copy
                # with the higher confidence (dedupe_detections, postprocess.py:146); on a
                # tie, the lower subregion's. In aca_full_2026-10-23 the two copies are
                # identical in every column (1 Nov), so the rule only matters if they ever
                # differ; the record counts how often they do.
                twice = cols["scored"][row]
                if twice.any():
                    old = np.column_stack([cols[c][row[twice]] for c in values]).astype(np.float64)
                    now = np.column_stack([new[c][twice] for c in values]).astype(np.float64)
                    same = (old == now) | (np.isnan(old) & np.isnan(now))
                    border["chips_scored_twice"] += int(twice.sum())
                    border["copies_differ"] += int((~same.all(axis=1)).sum())
                    border["largest_confidence_gap"] = max(
                        border["largest_confidence_gap"],
                        float(np.abs(old[:, 0] - now[:, 0]).max()))
                take = ~twice | (new["confidence"] > cols["confidence"][row])
                for c in values:
                    cols[c][row[take]] = new[c][take]
                cols["subregion"][row[take]] = sub
                cols["scored"][row] = True
                rows += len(chip_id)
        if rows != found[sub]["status"]["chips_saved"]:
            raise Broken(f"subregion {sub}'s part files hold {rows:,} chips but its "
                         f"_status.json says {found[sub]['status']['chips_saved']:,}")

    # A chip without a score must belong to a tile that failed every try; anything else
    # means the part files and _status.json disagree. It keeps its row (P1), credited to
    # its tile's subregion.
    unscored = ~cols["scored"]
    failed = {tile_number(k) for f in found.values() for k in f["status"]["tiles_failed"]}
    if set(np.unique(plan.ids[unscored] // 1000).tolist()) - failed:
        raise Broken("some tiles have no scores but are not listed as failed")
    cols["subregion"][unscored] = plan.home[unscored]
    return cols, border


def mark_lost_quality(cols: dict[str, np.ndarray]) -> dict:
    """Quality values that masked pixels made wrong become missing; returns counts."""
    # ADDITION: a pixel with no clear image of the period is masked by Earth Engine. Since
    # detect.py asks for 32-bit pixels (22 Oct) such a pixel arrives as -inf, not 0, and
    # detect.py took it for a seen pixel (fixed in detect.py, quality(), on 1 Nov). In
    # every chip with such a pixel this left empty_share at 0, ndvi_mean and bsi_mean
    # missing, and cirrus_mean and cs_clear_mean at -inf. The scores are not affected: the
    # model's input is clipped to 0..1, so the pixel reaches it as 0, as in AMW. What
    # can't be recovered becomes missing; nothing is guessed. On part files written after
    # the fix there is no -inf, and this changes nothing.
    if "cs_clear_mean" not in cols:                # runs before 23 Oct: no such columns
        return {"chips_with_lost_quality_values": 0}
    gaps = np.isneginf(cols["cs_clear_mean"])      # some, not all, pixels without a clear image
    empty = cols["n_clear"] == 0                   # no clear image in any pixel of the chip
    cols["empty_share"][gaps] = np.nan             # unknown: somewhere between 0 and 1
    cols["empty_share"][empty] = 1.0               # known exactly
    for c in ("cs_clear_mean", "ndvi_mean", "bsi_mean", "cirrus_mean"):
        cols[c][~np.isfinite(cols[c])] = np.nan
    return {"chips_with_lost_quality_values": int(gaps.sum()),
            "chips_without_any_clear_image": int(empty.sum())}


def scores_table(plan: Plan, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    """The scores layer as a table, without copying the arrays."""
    order = ["subregion", "scored", "confidence", *VOTES,
             *[c for c in QUALITY if c in cols]]
    return pd.DataFrame({"chip_id": plan.ids, **{c: cols[c] for c in order}}, copy=False)


# ------------------------------------------------------------ layer 2: postprocess

def kth_neighbor_km_geodesic(lon: np.ndarray, lat: np.ndarray, k: int) -> np.ndarray:
    """Distance (km) from each point to its k-th nearest other point, on the Earth itself."""
    # ADDITION (open decision D3): AMW measures in one UTM zone for the whole file
    # (postprocess.py:190), which reads distances up to ~5% too long at Amazon_ACA's
    # ends. The neighbours are found on a sphere (haversine), and the k-th one's distance
    # is then measured on the WGS84 ellipsoid.
    out = np.full(len(lon), np.nan, dtype=np.float64)
    if len(lon) <= k:                             # Same as AMW: too few points, no distance
        return out
    points = np.radians(np.column_stack([lat, lon]))
    _, nearest = BallTree(points, metric="haversine").query(points, k=k + 1)
    other = nearest[:, k]                         # column 0 is the point itself
    _, _, metres = GEOD.inv(lon, lat, lon[other], lat[other])
    out[:] = metres / 1000.0
    return out


def postprocess_layer(plan: Plan, scored: np.ndarray, conf: np.ndarray,
                      cfg: Config) -> tuple[pd.DataFrame, dict]:
    """AMW's detection flags for every chip, with the distances they rest on, and counts."""
    th = cfg.thresholds
    n = len(plan.ids)
    out = {"chip_id": plan.ids}
    counts = {}

    def flag(values: np.ndarray) -> pd.arrays.BooleanArray:
        """True/False for scored chips, missing (<NA>) for chips without a score (P1)."""
        return pd.arrays.BooleanArray(values, ~scored)

    # Same as AMW: a raw detection is a mean score strictly above 0.4
    # (inference_engine.py:803), and an Andes detection strictly above 0.2 on the Andes
    # chips. DEVIATION from AMW: kept as flags on every chip, not as files of positives.
    out["det_raw"] = flag(scored & (conf > th.raw))
    out["det_andes"] = flag(scored & plan.in_andes & (conf > th.andes))

    # The chips any of the two catalogues can contain, and their squares from the plan.
    low = min(th.loose.t_main, th.stringent.t_main)
    candidates = np.flatnonzero(scored & (conf >= low))
    squares = plan.squares(plan.ids[candidates])

    for name, pair in (("loose", th.loose), ("stringent", th.stringent)):
        kth = np.full(n, np.nan)                   # ADDITION: the distances the flag rests
        kth_geo = np.full(n, np.nan)               # on, so a better rule can be derived later
        detected = np.zeros(n, dtype=bool)         # (open decisions D1, D3); empty outside
        counts[name] = {"catalogue": 0, "twins_dropped": 0, "isolated": 0, "detected": 0}

        # Same as AMW: the neighbour catalogue is every chip at or above t_main, across all
        # subregions of the period (postprocess.py:362). AMW runs it on the raw detections
        # (> 0.4); t_main is above 0.4, so the catalogue is the same.
        mine = conf[candidates] >= pair.t_main
        if mine.any():
            box = squares[mine]
            catalogue = gpd.GeoDataFrame(
                {"confidence": conf[candidates[mine]]},
                geometry=shapely.box(box[:, 0], box[:, 1], box[:, 2], box[:, 3]),
                crs="EPSG:4326")
            # Same as AMW: one chip per location before the neighbour search; where two
            # chips share AMW's rounded centre, the higher score stays (dedupe_detections,
            # postprocess.py:146). In aca_full_2026-10-23 that happens only on the UTM zone
            # seam at 60°W: 368 chip pairs of subregion 2 (open decision D5). Both chips
            # keep their rows; the one dropped gets no distance and det = False.
            kept = amw.dedupe_detections(catalogue, verbose=False)
            rows = candidates[mine][kept.index.to_numpy()]
            dist = amw.kth_neighbor_km_on_catalog(amw.centroids_xy_m(kept),
                                                  k=th.neighbours_k)   # AMW: postprocess.py:365-366
            keep = amw.keep_dual_threshold(conf[rows], dist, t_main=pair.t_main,
                                           t_iso=pair.t_iso,
                                           isolation_km=th.isolation_km)  # AMW: postprocess.py:369
            detected[rows] = keep
            kth[rows] = dist
            b = box[kept.index.to_numpy()]
            kth_geo[rows] = kth_neighbor_km_geodesic((b[:, 0] + b[:, 2]) / 2,
                                                     (b[:, 1] + b[:, 3]) / 2, th.neighbours_k)
            isolated = ~(np.isfinite(dist) & (dist <= th.isolation_km))   # AMW's rule, incl. NaN
            counts[name] = {"catalogue": len(rows), "twins_dropped": int(mine.sum()) - len(rows),
                            "isolated": int(isolated.sum()), "detected": int(keep.sum())}
        out[f"kth_km_{name}"] = kth
        out[f"kth_km_geo_{name}"] = kth_geo
        out[f"det_{name}"] = flag(detected)        # Same as AMW: nothing in the catalogue,
                                                   # nothing detected (postprocess.py:362)
    counts["raw"] = int(out["det_raw"].sum())
    counts["andes"] = int(out["det_andes"].sum())
    order = ["chip_id", "det_raw", "det_andes", "kth_km_loose", "kth_km_geo_loose", "det_loose",
             "kth_km_stringent", "kth_km_geo_stringent", "det_stringent"]
    return pd.DataFrame({c: out[c] for c in order}, copy=False), counts


# ------------------------------------------------------------ writing

def layer_folder(panel_dir: Path, frequency: str, layer: str, period: Period) -> Path:
    return panel_dir / frequency / layer / f"period={period.tag}"


def write_layer(folder: Path, table: pd.DataFrame, record: dict, lock: FolderLock) -> None:
    """Write one layer of one period, all or nothing: the data first, then its record."""
    folder.mkdir(parents=True, exist_ok=True)
    lock.check()                                  # still ours? never write into a stolen folder
    tmp = folder / "_data.tmp"                    # "_": readers skip it until it is complete
    table.to_parquet(tmp, index=False, compression="zstd", row_group_size=ROW_GROUP_ROWS)
    os.replace(tmp, folder / "data.parquet")
    tmp = folder / "_record.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, folder / "_record.json")      # the record last: it means "complete"


def assemble(cfg: Config, run_dir: Path, panel_dir: Path, frequency: str, period: Period,
             plan: Plan, others: list[int], allow_failed: bool) -> str:
    """Assemble one period's missing layers. Returns a line for the log."""
    found = detect_inputs(run_dir, frequency, period, plan.subregions, allow_failed)
    inputs = inputs_summary(found)
    # One writer per period: its lock lives in the first layer's folder (lock.py).
    with FolderLock(layer_folder(panel_dir, frequency, "scores", period)) as lock:
        return assemble_locked(cfg, panel_dir, frequency, period, found, inputs, plan,
                               others, lock)


def assemble_locked(cfg: Config, panel_dir: Path, frequency: str, period: Period,
                    found: dict[int, dict], inputs: dict, plan: Plan, others: list[int],
                    lock: FolderLock) -> str:
    """The rest of assemble(), while holding the period's lock."""
    todo = []
    for name in LAYERS:
        record = layer_folder(panel_dir, frequency, name, period) / "_record.json"
        if not record.exists():
            todo.append(name)
        elif json.loads(record.read_text())["inputs"] != inputs:
            raise NotReady(f"detect's output changed after the {name} layer was written. "
                           f"The panel is never overwritten: move "
                           f"{panel_dir.name}/{frequency}/*/period={period.tag} aside by hand first")
    if not todo:
        return f"{period.tag}  already assembled"

    failed_keys = sorted({k for f in found.values() for k in f["status"]["tiles_failed"]})
    about = {"frequency": frequency, "period": period.tag, "assembled_utc": utc_now(),
             "subregions": plan.subregions, "inputs": inputs,
             "model_members": next(iter(found.values()))["status"]["model_members"],
             "rows": len(plan.ids), "tiles_failed": failed_keys} | environment(cfg)
    if others:                                    # a panel of part of the region
        about["note"] = (f"Only subregions {plan.subregions}: neighbours in subregions "
                         f"{others} are missing, so the isolation flags of chips near "
                         f"those borders can differ from the whole region's panel.")
    scores_path = layer_folder(panel_dir, frequency, "scores", period) / "data.parquet"
    line = []

    if "scores" in todo:
        cols, border = place_scores(found, plan)
        lost = mark_lost_quality(cols)
        table = scores_table(plan, cols)
        record = about | {"layer": "scores", "columns": list(table.columns),
                          "scored": int(cols["scored"].sum()),
                          "border": {"tiles": plan.border_tiles, **border},
                          "masked_pixels": lost,
                          "mean_n_clear": float(np.nanmean(cols["n_clear"])),
                          "share_n_clear_below_1": float((cols["n_clear"] < 1).mean())}
        write_layer(scores_path.parent, table, record, lock)
        del cols, table                           # free the big arrays before layer 2
        line.append(f"{record['scored']:,} of {len(plan.ids):,} chips scored, "
                    f"{border['chips_scored_twice']:,} border chips "
                    f"({border['copies_differ']:,} copies differ)")

    if "postprocess" in todo:
        # Layer 2 is computed from layer 1 as written, never from anything else.
        s = pq.read_table(scores_path, columns=["chip_id", "scored", "confidence"])
        if not np.array_equal(s.column("chip_id").to_numpy(), plan.ids):
            raise Broken("the scores layer's rows don't match the plan")
        flags, counts = postprocess_layer(plan, s.column("scored").to_numpy(),
                                          s.column("confidence").to_numpy(), cfg)
        th = cfg.thresholds
        record = about | {"layer": "postprocess", "columns": list(flags.columns),
                          "thresholds": {"raw": th.raw, "andes": th.andes,
                                         "loose": asdict(th.loose),
                                         "stringent": asdict(th.stringent),
                                         "neighbours_k": th.neighbours_k,
                                         "isolation_km": th.isolation_km},
                          "counts": counts}
        write_layer(layer_folder(panel_dir, frequency, "postprocess", period), flags,
                    record, lock)
        line.append(f"detected raw {counts['raw']:,}, loose {counts['loose']['detected']:,}, "
                    f"stringent {counts['stringent']['detected']:,}, Andes {counts['andes']:,}")
    return f"{period.tag}  " + "; ".join(line) + f"; wrote {', '.join(todo)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Assemble periods of the panel from detect's "
                                             "part files.")
    ap.add_argument("--run", required=True, help="the run folder made by plan.py")
    ap.add_argument("--frequency", required=True, choices=("monthly", "quarterly"))
    ap.add_argument("--subregions", type=int, nargs="+",
                    help="only these subregions, e.g. --subregions 2 3 (default: all)")
    ap.add_argument("--from", dest="first", help="first period, e.g. 2018-01 or 2018Q1")
    ap.add_argument("--to", dest="last", help="last period")
    ap.add_argument("--allow-failed", action="store_true",
                    help="assemble periods with tiles that failed every try (their chips "
                         "get empty scores and scored = False)")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()      # as detect.py reads --run
    if not (run_dir / "run" / "config.yaml").is_file():
        raise SystemExit(f"ASSEMBLE ERROR: {run_dir} is not a planned run")
    try:
        cfg = load_config(run_dir / "run" / "config.yaml")   # the run's copy, never the live one
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")
    if args.frequency not in cfg.periods:
        raise SystemExit(f"ASSEMBLE ERROR: this run has no {args.frequency} periods")
    periods = pick_periods(cfg, args.frequency, args.first, args.last)

    tiles = pd.read_csv(run_dir / "plan" / "tiles.csv")
    planned = sorted(tiles["subregion"].unique().tolist())   # those with tiles
    chosen = sorted(set(args.subregions or planned))
    if set(chosen) - set(planned):
        raise SystemExit(f"ASSEMBLE ERROR: the plan has no tiles in subregion(s) "
                         f"{sorted(set(chosen) - set(planned))}; it has {planned}")
    others = [n for n in planned if n not in chosen]
    panel_dir = run_dir / panel_name(chosen if others else None)
    print(f"reading the plan: subregions {chosen} -> {panel_dir.name}/{args.frequency}/",
          flush=True)
    plan = Plan(run_dir, tiles, chosen)

    waiting: dict[str, list[str]] = {}            # reason -> the periods it holds back
    broken = []
    for period in periods:
        try:
            print(assemble(cfg, run_dir, panel_dir, args.frequency, period, plan, others,
                           args.allow_failed), flush=True)
        except (NotReady, Locked) as why:
            waiting.setdefault(str(why), []).append(period.tag)
        except Broken as what:                    # report it, and go on with the others
            print(f"{period.tag}  ASSEMBLE ERROR: {what}", flush=True)
            broken.append(period.tag)
    problems = [f"  {len(tags)} period(s), {tags[0]}"
                + (f" to {tags[-1]}" if len(tags) > 1 else "") + f": {why}"
                for why, tags in waiting.items()]    # one line per reason, not per period
    if broken:
        problems.append(f"  {len(broken)} period(s) with errors (above): {', '.join(broken)}")
    if problems:
        raise SystemExit("not assembled:\n" + "\n".join(problems))
