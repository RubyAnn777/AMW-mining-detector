"""Put one period's scores together across subregions, and add AMW's postprocess flags.

Run after detect.py has finished every subregion of a period:
    python panel_pipeline/assemble_period.py --run tests_output/<run> --frequency monthly
    ... --from 2026-01 --to 2026-03   only these periods (default: all of the run's)
    ... --allow-failed                accept tiles that failed every try (their chips get
                                      empty scores, scored = False)

Writes two layers of the panel per period. Both hold every chip of the plan, once, sorted
by chip_id, so their rows line up one to one:
    <run>/panel/<frequency>/scores/period=<tag>/data.parquet        detect's values
    <run>/panel/<frequency>/postprocess/period=<tag>/data.parquet   AMW's flags, distances
and a _record.json next to each: what went in, counts, settings, which code ran.

pd.read_parquet("<run>/panel/monthly/scores") reads every period as one table, with the
folder name as a `period` column; files starting with _ are skipped.

Nothing written is ever changed (open_design_decisions.Rmd, P1 and P2). A layer that
exists is skipped; later steps add new layers next to these. Run the same command again
at any time: it assembles what is ready and missing, and says why the rest isn't.

Memory: one period at a time, whole. That is ~40 MB for a test run, and roughly 10-20 GB
for a full Amazon_ACA period: fine on a cluster node, too much for a laptop.
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

# AMW code. This must come after `grid`, which puts AMW's core/ on the search path.
import postprocess as amw

LAYERS = ("scores", "postprocess")
VOTES = [f"vote_{i}" for i in range(1, 7)]
ROW_GROUP_ROWS = 200 * 529     # 200 tiles of 529 chips. build_panel.py will read an area
                               # (a chip_id range) at a time; Parquet skips the row groups
                               # outside it, because each records its smallest and largest id
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

def read_scores(found: dict[int, dict]) -> pd.DataFrame:
    """Every subregion's part files as one table, with the subregion as a column."""
    frames = []
    for n, f in found.items():
        parts = sorted(f["folder"].glob("part_*.parquet"))
        rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if len(rows) != f["status"]["chips_saved"]:
            raise Broken(f"subregion {n}'s part files hold {len(rows):,} chips but its "
                         f"_status.json says {f['status']['chips_saved']:,}")
        frames.append(rows.assign(subregion=np.int8(n)))
    return pd.concat(frames, ignore_index=True)


def scores_layer(scores: pd.DataFrame, plan_ids: np.ndarray, tiles: pd.DataFrame,
                 failed_keys: set[str]) -> pd.DataFrame:
    """One row per planned chip, in chip_id order; chips of failed tiles keep an empty row."""
    if scores["chip_id"].duplicated().any():
        # Open decision D7: a tile on a subregion border is scored under both
        # subregions. AMW keeps the copy with the higher confidence (postprocess.py:146);
        # not built yet, so stop instead of guessing.
        n = int(scores["chip_id"].duplicated().sum())
        raise Broken(f"{n:,} chips were scored twice (border tiles, open decision D7); "
                     f"the step that resolves them isn't built yet")
    if not scores["chip_id"].isin(plan_ids).all():
        raise Broken("the part files hold chips that aren't in the plan")

    # ADDITION (P1, a complete panel): AMW keeps only chips above 0.4. Every planned chip
    # gets a row here, so "no data" can never be read as "no mine".
    layer = pd.DataFrame({"chip_id": plan_ids}).merge(
        scores, on="chip_id", how="left", validate="one_to_one", indicator=True)
    layer["scored"] = layer.pop("_merge").eq("both")

    # A chip without a score must belong to a tile that failed every try; anything else
    # means the part files and _status.json disagree.
    unscored_tiles = set(np.unique(layer.loc[~layer["scored"], "chip_id"] // 1000).tolist())
    failed_tiles = {tile_number(k) for k in failed_keys}
    if unscored_tiles - failed_tiles:
        raise Broken(f"{len(unscored_tiles - failed_tiles)} tile(s) have no scores but are "
                     f"not listed as failed")
    if not layer["scored"].all():                 # the subregion of an unscored chip
        where = dict(zip(tiles["tile_key"].map(tile_number), tiles["subregion"]))
        missing = ~layer["scored"]
        layer.loc[missing, "subregion"] = (layer.loc[missing, "chip_id"] // 1000).map(where)

    return layer.astype({"subregion": "int8", "confidence": "float64",
                         **{c: "float32" for c in VOTES + ["n_obs", "n_clear", "empty_share"]}})[
        ["chip_id", "subregion", "scored", "confidence", *VOTES,
         "n_obs", "n_clear", "empty_share"]]


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


def postprocess_layer(layer: pd.DataFrame, chips: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """AMW's detection flags for every chip, with the distances they rest on, and counts."""
    th = cfg.thresholds
    scored = layer["scored"].to_numpy()
    conf = layer["confidence"].to_numpy(dtype=np.float64)    # NaN where not scored
    out = pd.DataFrame({"chip_id": layer["chip_id"]})
    counts = {}

    def flag(values: np.ndarray) -> pd.Series:
        """True/False for scored chips, missing (<NA>) for chips without a score (P1)."""
        return pd.Series(values, dtype="boolean").mask(~scored)

    # Same as AMW: a raw detection is a mean score strictly above 0.4
    # (inference_engine.py:803), and an Andes detection strictly above 0.2 on the Andes
    # chips. DEVIATION from AMW: kept as flags on every chip, not as files of positives.
    out["det_raw"] = flag(conf > th.raw)
    out["det_andes"] = flag(chips["in_andes"].to_numpy() & (conf > th.andes))

    centre_lon = ((chips["minx"] + chips["maxx"]) / 2).to_numpy()
    centre_lat = ((chips["miny"] + chips["maxy"]) / 2).to_numpy()
    for name, pair in (("loose", th.loose), ("stringent", th.stringent)):
        # Same as AMW: the neighbour catalogue is every chip at or above t_main, across all
        # subregions of the period (postprocess.py:362). AMW runs it on the raw detections
        # (> 0.4); t_main is above 0.4, so the catalogue is the same.
        catalogue = scored & (conf >= pair.t_main)
        out[f"kth_km_{name}"] = np.nan             # ADDITION: the distances the flag rests on,
        out[f"kth_km_geo_{name}"] = np.nan         # so a better rule can be derived later
                                                   # (open decisions D1, D3); empty outside
                                                   # the catalogue
        if not catalogue.any():                    # Same as AMW: nothing to filter
            out[f"det_{name}"] = flag(np.zeros(len(out), dtype=bool))  # (postprocess.py:362)
            counts[name] = {"catalogue": 0, "isolated": 0, "detected": 0}
            continue
        squares = gpd.GeoDataFrame(geometry=shapely.box(
            chips.loc[catalogue, "minx"], chips.loc[catalogue, "miny"],
            chips.loc[catalogue, "maxx"], chips.loc[catalogue, "maxy"]), crs="EPSG:4326")
        kth = amw.kth_neighbor_km_on_catalog(amw.centroids_xy_m(squares),
                                             k=th.neighbours_k)       # AMW: postprocess.py:365-366
        keep = amw.keep_dual_threshold(conf[catalogue], kth, t_main=pair.t_main,
                                       t_iso=pair.t_iso,
                                       isolation_km=th.isolation_km)  # AMW: postprocess.py:369
        detected = np.zeros(len(out), dtype=bool)
        detected[catalogue] = keep
        out[f"det_{name}"] = flag(detected)

        out.loc[catalogue, f"kth_km_{name}"] = kth
        out.loc[catalogue, f"kth_km_geo_{name}"] = kth_neighbor_km_geodesic(
            centre_lon[catalogue], centre_lat[catalogue], th.neighbours_k)

        isolated = ~(np.isfinite(kth) & (kth <= th.isolation_km))     # AMW's rule, incl. NaN
        counts[name] = {"catalogue": int(catalogue.sum()),
                        "isolated": int(isolated.sum()),
                        "detected": int(keep.sum())}
    counts["raw"] = int(out["det_raw"].sum())
    counts["andes"] = int(out["det_andes"].sum())
    return out, counts


# ------------------------------------------------------------ writing

def layer_folder(run_dir: Path, frequency: str, layer: str, period: Period) -> Path:
    return run_dir / "panel" / frequency / layer / f"period={period.tag}"


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


def assemble(cfg: Config, run_dir: Path, frequency: str, period: Period,
             subregions: list[int], tiles: pd.DataFrame, chips: pd.DataFrame,
             allow_failed: bool) -> str:
    """Assemble one period's missing layers. Returns a line for the log."""
    found = detect_inputs(run_dir, frequency, period, subregions, allow_failed)
    inputs = inputs_summary(found)
    # One writer per period: its lock lives in the first layer's folder (lock.py).
    with FolderLock(layer_folder(run_dir, frequency, "scores", period)) as lock:
        return assemble_locked(cfg, run_dir, frequency, period, found, inputs, tiles,
                               chips, lock)


def assemble_locked(cfg: Config, run_dir: Path, frequency: str, period: Period,
                    found: dict[int, dict], inputs: dict, tiles: pd.DataFrame,
                    chips: pd.DataFrame, lock: FolderLock) -> str:
    """The rest of assemble(), while holding the period's lock."""
    todo = []
    for name in LAYERS:
        record = layer_folder(run_dir, frequency, name, period) / "_record.json"
        if not record.exists():
            todo.append(name)
        elif json.loads(record.read_text())["inputs"] != inputs:
            raise NotReady(f"detect's output changed after the {name} layer was written. "
                           f"The panel is never overwritten: move "
                           f"panel/{frequency}/*/period={period.tag} aside by hand first")
    if not todo:
        return f"{period.tag}  already assembled"

    failed_keys = {k for f in found.values() for k in f["status"]["tiles_failed"]}
    scores = scores_layer(read_scores(found), chips["chip_id"].to_numpy(), tiles, failed_keys)
    flags, counts = postprocess_layer(scores, chips, cfg)

    about = {"frequency": frequency, "period": period.tag, "assembled_utc": utc_now(),
             "inputs": inputs,
             "model_members": next(iter(found.values()))["status"]["model_members"],
             "rows": len(scores), "scored": int(scores["scored"].sum()),
             "tiles_failed": sorted(failed_keys)} | environment(cfg)
    records = {
        "scores": about | {"layer": "scores", "columns": list(scores.columns),
                           "mean_n_clear": float(scores["n_clear"].mean()),
                           "share_n_clear_below_1": float((scores["n_clear"] < 1).mean())},
        "postprocess": about | {"layer": "postprocess", "columns": list(flags.columns),
                                "thresholds": {"raw": cfg.thresholds.raw,
                                               "andes": cfg.thresholds.andes,
                                               "loose": asdict(cfg.thresholds.loose),
                                               "stringent": asdict(cfg.thresholds.stringent),
                                               "neighbours_k": cfg.thresholds.neighbours_k,
                                               "isolation_km": cfg.thresholds.isolation_km},
                                "counts": counts},
    }
    tables = {"scores": scores, "postprocess": flags}
    for name in todo:
        write_layer(layer_folder(run_dir, frequency, name, period), tables[name],
                    records[name], lock)
    return (f"{period.tag}  {len(scores):,} chips, {records['scores']['scored']:,} scored, "
            f"{len(failed_keys)} failed tiles; detected raw {counts['raw']:,}, "
            f"loose {counts['loose']['detected']:,}, stringent "
            f"{counts['stringent']['detected']:,}, Andes {counts['andes']:,}; "
            f"wrote {', '.join(todo)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Assemble periods of the panel from detect's "
                                             "part files.")
    ap.add_argument("--run", required=True, help="the run folder made by plan.py")
    ap.add_argument("--frequency", required=True, choices=("monthly", "quarterly"))
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
    subregions = sorted(tiles["subregion"].unique().tolist())   # those with tiles
    chips = pd.read_parquet(run_dir / "plan" / "chips.parquet",
                            columns=["chip_id", "minx", "miny", "maxx", "maxy", "in_andes"])

    waiting: dict[str, list[str]] = {}            # reason -> the periods it holds back
    broken = []
    for period in periods:
        try:
            print(assemble(cfg, run_dir, args.frequency, period, subregions, tiles, chips,
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
