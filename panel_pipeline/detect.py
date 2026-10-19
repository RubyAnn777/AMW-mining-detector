"""Score every chip of one subregion with AMW's model, one period at a time.

Run after plan.py, once per frequency and subregion:
    python panel_pipeline/detect.py --run tests_output/pilot --frequency monthly --subregion 6
    ... --from 2018-01 --to 2018-12   only these periods (default: all of the run's)
    ... --limit 1                     only the first tile: one Earth Engine request

Writes <run>/detect/<frequency>/period=<period>/subregion=<n>/
    part_00001.parquet, ...   one row per chip: chip_id, confidence, six votes, image counts
    _status.json              how far it got, which tiles failed and why, what code ran

pd.read_parquet("<run>/detect/monthly") reads every part as one table, with the
folder names as two extra columns (period, subregion); files starting with _ are skipped.

Run the same command again to resume: saved tiles are skipped, failed ones retried.
Never run two copies on the same frequency and subregion at once.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from descarteslabs.geo import DLTile

from config import Config, load_config
from grid import CHIP_PX, chip_ids
from periods import Period
from provenance import environment, repo_path, sha256

# AMW code. These must come after `grid`, which puts AMW's core/ on the search path.
import gee
from tf_darwin import apply_darwin_tf_compat
from tile_utils import cut_chips

SPECTRAL = list(gee.DataConfig._BAND_IDS["S2L1C"])   # AMW's 13 bands, model's order (gee.py:134)
COUNTS = ["n_obs", "n_clear"]                         # ADDITION: see PanelExtractor


# ------------------------------------------------------------ imagery

class PanelExtractor(gee.GEE_Data_Extractor):
    """AMW's Earth Engine downloader, plus two bands that count images per pixel."""

    def _build_composite(self, start_date, end_date):
        composite = super()._build_composite(start_date, end_date)   # AMW's median, unchanged
        # ADDITION: per pixel, the images AMW's recipe could use (n_obs: those with a
        # cloud score, cloudy or clear) and those passing its cloud test (n_clear: what
        # the median is made of). Same collections, dates and test as gee.py:208-229.
        scores = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                  .filterDate(start_date, end_date)
                  .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"),
                                  ["cs_cdf"])
                  .select("cs_cdf"))
        clear = scores.map(lambda img: img.updateMask(img.gte(self.config.clear_threshold)))
        return (composite.addBands(scores.count().rename("n_obs"))
                         .addBands(clear.count().rename("n_clear")))


def extractor_for(cfg: Config, period: Period) -> PanelExtractor:
    """AMW's downloader for one period, set up from the run's settings."""
    data_config = gee.DataConfig(
        tilesize=cfg.imagery.tilesize,
        pad=cfg.imagery.pad,
        collection=cfg.imagery.collection,
        bands=SPECTRAL + COUNTS,      # ADDITION: the counts come in the same request
        clear_threshold=cfg.imagery.clear_threshold,
        max_workers=cfg.imagery.max_workers,
        # DEVIATION from AMW: no image cache (AMW passes --image_cache_dir, pipeline.py:161).
        # Ours would hold roughly 200-300 TB, and AMW's stores whole numbers (gee.py:493),
        # so a rerun from it scores slightly different pixels. Every score here uses the
        # pixels exactly as downloaded, like AMW's first pass; resuming avoids re-downloads.
        image_cache_dir=None,
    )
    # Dates as text, like AMW's driver. Earth Engine excludes the end date (Period.end).
    return PanelExtractor(period.start.isoformat(), period.end.isoformat(), data_config)


# ------------------------------------------------------------ scoring one tile

def score_tile(tile, pixels: np.ndarray, model, stride: int) -> pd.DataFrame:
    """One row per chip: AMW's confidence, the six votes behind it, the image counts."""
    spectral, counts = pixels[..., :len(SPECTRAL)], pixels[..., len(SPECTRAL):]
    bounds = tile.geometry.bounds

    # Same as AMW, line for line (inference_engine.py:945-949, 766, 786).
    normed = np.clip(spectral / 10000.0, 0, 1).astype(np.float32)
    chips, _ = cut_chips(normed, bounds, CHIP_PX, stride, crs="epsg:4326")
    votes = model.predict(chips, verbose=0)                 # one column per ensemble member
    confidence = np.asarray(np.mean(votes, axis=1), dtype=np.float64)

    # DEVIATION from AMW: AMW keeps only chips above a threshold (0.4, or 0.2 for the
    # Andes: pipeline.py:161 and :167, applied at inference_engine.py:803), with their
    # votes as text (:830). We keep every chip, so thresholds are applied later and
    # nothing is lost. confidence stays float64, AMW's exact number, so a test like
    # > 0.4 gives AMW's answer (as float32, a score of exactly 0.4000000060 would fail it).
    rows = pd.DataFrame({"chip_id": chip_ids(tile, len(chips)), "confidence": confidence})
    for i in range(votes.shape[1]):
        rows[f"vote_{i + 1}"] = votes[:, i]

    # ADDITION: per chip, the mean image counts and the share of empty pixels (no clear
    # image: 0 in every band, which the model sees as black), over the same squares.
    empty = (spectral == 0).all(axis=-1)
    layers, _ = cut_chips(np.dstack([counts, empty]), bounds, CHIP_PX, stride,
                          crs="epsg:4326")
    means = layers.mean(axis=(1, 2))                        # (chips, 3)
    rows["n_obs"] = means[:, 0]
    rows["n_clear"] = means[:, 1]
    rows["empty_share"] = means[:, 2]
    return rows


# ------------------------------------------------------------ one batch of tiles

def score_batch(extractor, tiles: list, model, stride: int):
    """Download tiles in parallel threads; score each here as soon as it arrives.

    Returns the scored tiles, and the error for each tile that failed.
    """
    # DEVIATION from AMW, in structure only: AMW starts all 500 downloads of a batch at
    # once and passes them to a scoring thread through a queue (inference_engine.py:
    # 1167-1211), so every downloaded tile stays in memory until the batch ends (about
    # 9 GB). Here this thread scores, and at most two downloads per worker are under
    # way or waiting, so memory stays small. Same downloads, same scores.
    workers = extractor.config.max_workers
    waiting = iter(tiles)
    running = {}                                 # download under way -> its tile
    frames, failed = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            for tile in itertools.islice(waiting, 2 * workers - len(running)):
                running[pool.submit(extractor.get_tile_data, tile)] = tile  # AMW: gee.py:289
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                tile = running.pop(future)
                try:
                    frames.append(score_tile(tile, future.result(), model, stride))
                except Exception as err:         # as AMW: note it, retry it in the next try
                    failed[tile.key] = f"{type(err).__name__}: {err}"
    return frames, failed


# ------------------------------------------------------------ the period's folder

def tile_number(tile) -> int:
    """A tile's chip_ids without the last three digits (the chip), see grid.py."""
    return int(chip_ids(tile, 1)[0]) // 1000


def saved_tiles(folder: Path) -> tuple[set[int], int]:
    """The tiles already saved in this folder's part files, and how many chips."""
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        return set(), 0
    ids = np.concatenate([pd.read_parquet(p, columns=["chip_id"])["chip_id"].to_numpy()
                          for p in parts])
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"DETECT ERROR: a chip is saved twice in {folder}; "
                         f"were two copies of detect.py writing here at once?")
    return set(np.unique(ids // 1000).tolist()), len(ids)


def write_part(folder: Path, frames: list[pd.DataFrame]) -> None:
    """Save a batch as the next numbered part file, all or nothing."""
    rows = pd.concat(frames, ignore_index=True).sort_values("chip_id", ignore_index=True)
    numbers = [int(p.stem.removeprefix("part_")) for p in folder.glob("part_*.parquet")]
    path = folder / f"part_{max(numbers, default=0) + 1:05d}.parquet"
    tmp = folder / f"_{path.stem}.tmp"           # "_": readers skip it until it is complete
    rows.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)                        # the part appears only when complete


def write_status(folder: Path, about: dict, planned: list, errors: dict,
                 sessions: list) -> dict:
    """Count what the part files hold, and write it to _status.json, all or nothing."""
    # ADDITION: AMW only logs how many tiles failed (inference_engine.py:1241). We record
    # which ones and why, and which code, packages and machine scored the period.
    saved, n_chips = saved_tiles(folder)
    missing = {t.key for t in planned if tile_number(t) not in saved}
    failed = {key: err for key, err in errors.items() if key in missing}
    status = about | {
        "complete": not missing,
        "tiles_planned": len(planned),
        "tiles_saved": len(planned) - len(missing),
        "chips_saved": n_chips,
        "tiles_not_tried": len(missing) - len(failed),
        "tiles_failed": failed,
        "sessions": sessions,
    }
    tmp = folder / "_status.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, folder / "_status.json")
    return status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ one period

def run_period(cfg: Config, period: Period, subregion: int, planned: list,
               limit: int | None, model, members: list[str], folder: Path) -> dict:
    """Score a period's tiles batch by batch, with AMW's retries, and record the outcome."""
    folder.mkdir(parents=True, exist_ok=True)
    for leftover in folder.glob("_*.tmp"):
        leftover.unlink()                        # half-written by a crash; never valid
    old = (json.loads((folder / "_status.json").read_text())
           if (folder / "_status.json").exists() else {})
    errors = old.get("tiles_failed", {})         # tile key -> its latest error
    sessions = old.get("sessions", []) + [
        {"started_utc": utc_now(), "finished_utc": None, "limit": limit} | environment(cfg)]
    about = {
        "period": period.tag,
        "frequency": period.frequency,
        "subregion": subregion,
        "start_date": period.start.isoformat(),
        "end_date_excluded": period.end.isoformat(),
        "ee_project": cfg.imagery.ee_project,
        "model": repo_path(cfg.inference.model_path),
        "model_members": members,                # vote_1, vote_2, ... in this order
    }
    write_status(folder, about, planned, errors, sessions)   # so even a crash leaves a record

    work = planned[:limit]                       # --limit: the first tiles only
    started = time.time()
    extractor = extractor_for(cfg, period)
    stride = CHIP_PX // cfg.inference.stride_ratio           # AMW: inference_engine.py:756
    size = cfg.inference.batch_tiles
    saved, _ = saved_tiles(folder)
    # Same as AMW: `tries` passes, each retrying the tiles the one before could not
    # finish (inference_engine.py:1151-1157). DEVIATION from AMW: a rerun resumes from
    # the part files; AMW starts over and relies on its image cache to be quick.
    for attempt in range(1, cfg.inference.tries + 1):
        todo = [t for t in work if tile_number(t) not in saved]
        for start in range(0, len(todo), size):
            batch = todo[start:start + size]
            frames, failed = score_batch(extractor, batch, model, stride)
            if frames:
                write_part(folder, frames)
            saved |= {tile_number(t) for t in batch if t.key not in failed}
            errors |= failed
            print(f"  {period.tag}  try {attempt}: {len(saved):,}/{len(planned):,} tiles "
                  f"saved, {len(failed)} failed in this batch "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)

    sessions[-1]["finished_utc"] = utc_now()
    return write_status(folder, about, planned, errors, sessions)


# ------------------------------------------------------------ checks before scoring

def check_model(cfg: Config, run_dir: Path) -> None:
    """Refuse to score with a model file that changed since the run was planned."""
    planned = json.loads((run_dir / "run" / "provenance.json").read_text())["file_sha256"]
    path = cfg.inference.model_path
    if planned.get(repo_path(path)) != sha256(path):
        raise SystemExit(f"DETECT ERROR: {path.name} has changed since this run was "
                         f"planned; one run uses one model throughout")


def pick_periods(cfg: Config, frequency: str, first: str | None,
                 last: str | None) -> list[Period]:
    """The run's periods of one frequency, from `first` to `last` (None: all)."""
    periods = cfg.periods[frequency]
    lo = Period.parse(first) if first else periods[0]
    hi = Period.parse(last) if last else periods[-1]
    if lo.frequency != frequency or hi.frequency != frequency:
        raise SystemExit(f"DETECT ERROR: --from and --to must be {frequency} periods, "
                         f"like {periods[0].tag}")
    chosen = [p for p in periods if lo <= p <= hi]
    if not chosen:
        raise SystemExit(f"DETECT ERROR: this run has no {frequency} periods "
                         f"from {lo.tag} to {hi.tag}")
    return chosen


def subregion_tiles(run_dir: Path, subregion: int) -> list:
    """The subregion's tiles as planned (plan/tiles.csv), in AMW's order."""
    tiles = pd.read_csv(run_dir / "plan" / "tiles.csv")
    keys = tiles.loc[tiles["subregion"] == subregion, "tile_key"]
    return [DLTile.from_key(key) for key in keys]   # rebuilt from the key, no network


def member_names(model) -> list[str]:
    """The ensemble's members, in the order of its outputs (vote_1, vote_2, ...)."""
    joined = model.layers[-1]                    # the layer that sets the votes side by side
    return [layer.name for layer in joined.inbound_nodes[0].inbound_layers]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score every chip of one subregion "
                                             "with AMW's model.")
    ap.add_argument("--run", required=True, help="the run folder made by plan.py")
    ap.add_argument("--frequency", required=True, choices=("monthly", "quarterly"))
    ap.add_argument("--subregion", required=True, type=int)
    ap.add_argument("--from", dest="first", help="first period, e.g. 2018-01 or 2018Q1")
    ap.add_argument("--to", dest="last", help="last period")
    ap.add_argument("--limit", type=int, help="score only the first LIMIT tiles (a test)")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not (run_dir / "run" / "config.yaml").is_file():
        raise SystemExit(f"DETECT ERROR: {run_dir} is not a planned run; run plan.py first")
    try:
        cfg = load_config(run_dir / "run" / "config.yaml")   # the run's copy, never the live one
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")
    if args.frequency not in cfg.periods:
        raise SystemExit(f"DETECT ERROR: this run has no {args.frequency} periods")
    if args.subregion not in cfg.region.subregion_files:
        raise SystemExit(f"DETECT ERROR: this run has no subregion {args.subregion}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("DETECT ERROR: --limit must be at least 1")
    check_model(cfg, run_dir)
    periods = pick_periods(cfg, args.frequency, args.first, args.last)
    planned = subregion_tiles(run_dir, args.subregion)
    if not planned:
        raise SystemExit(f"DETECT ERROR: subregion {args.subregion} has no tiles in this "
                         f"run (a test run's clip_to can leave a subregion empty)")

    # DEVIATION from AMW: no separate Andes run. AMW scores the Andes tiles a second
    # time to keep chips above 0.2 (pipeline.py:163-167). Those tiles are all in the
    # subregion lists (plan.py checks) and we keep every score, so the Andes chips'
    # scores are already here; the 0.2 threshold is applied later, to in_andes chips.

    gee.EE_PROJECT = cfg.imagery.ee_project      # read by AMW's Earth Engine login (gee.py:73)
    gee._ensure_earth_engine_initialized()       # AMW's login (gee.py:47), first, so a login
                                                 # problem stops us before any folder is made
    apply_darwin_tf_compat()                     # AMW: inference_engine.py:48; macOS only
    model = tf.keras.models.load_model(str(cfg.inference.model_path),
                                       compile=False)          # AMW: inference_engine.py:424
    members = member_names(model)

    print(f"{len(planned):,} tiles in subregion {args.subregion}, {len(periods)} "
          f"{args.frequency} periods, {periods[0].tag} to {periods[-1].tag}"
          + (f", first {args.limit} tile(s) only" if args.limit else ""), flush=True)
    failed_periods = 0
    for period in periods:
        folder = (run_dir / "detect" / args.frequency / f"period={period.tag}"
                  / f"subregion={args.subregion}")
        status = run_period(cfg, period, args.subregion, planned, args.limit, model,
                            members, folder)
        print(f"{period.tag}  {status['tiles_saved']:,}/{status['tiles_planned']:,} tiles "
              f"saved, {len(status['tiles_failed'])} failed  ->  {folder}", flush=True)
        failed_periods += bool(status["tiles_failed"])
    if failed_periods:
        raise SystemExit(f"{failed_periods} period(s) have failed tiles (see _status.json); "
                         f"run the same command again to retry them")
