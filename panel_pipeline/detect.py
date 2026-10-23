"""Score every chip of one subregion with AMW's model, one period at a time.

Run after plan.py, once per frequency and subregion:
    python panel_pipeline/detect.py --run tests_output/pilot --frequency monthly --subregion 6
    ... --from 2018-01 --to 2018-12   only these periods (default: all of the run's)
    ... --limit 1                     only the first tile: one Earth Engine request
    ... --workers 16 --batch-tiles 100 --tries 2   run-time choices (default: the run's config)
    ... --flush-minutes 2             save a part at least this often (default 2)
    ... --mac-eager                   on macOS, AMW's slower TensorFlow setting (if crashes)
    ... --image-cache tests_output/image_cache     tests only: keep the downloaded images

Writes <run>/detect/<frequency>/period=<period>/subregion=<n>/
    part_00001.parquet, ...   one row per chip: chip_id, confidence, six votes, and the
                              quality columns (README A6);
                              a new part every --batch-tiles tiles or --flush-minutes,
                              whichever comes first. Once the period is complete, they
                              are merged into part_00000.parquet
    _status.json              how far it got, which tiles failed and why, what code ran;
                              updated with every part
    _log.txt                  the progress lines printed on screen, plus each failed or
                              slow tile and Earth Engine's retries
    _lock.json                only while a process writes here: which one (lock.py)

pd.read_parquet("<run>/detect/monthly") reads every part as one table, with the
folder names as two extra columns (period, subregion); files starting with _ are skipped.

Each progress line says how fast it goes and what limits it: "workers busy" near 100%
means Earth Engine is the limit (more --workers may help, up to the project's quota);
"scoring busy" near 100% means this computer's scoring is (more workers won't help).

Run the same command again to resume: saved tiles are skipped, failed ones retried.
Ctrl-C (or a scheduler's time limit) saves the tiles scored so far, gives the lock back
and stops; only the downloads under way are lost. A second Ctrl-C stops at once.
Each period's folder is locked while it is written (_lock.json, see lock.py): a second
copy of this command skips the periods another process is writing.
On a cluster, one task is one period (--from X --to X); status.py shows how far a run
has got and prints the commands still to run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import platform
import shutil
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import yaml
from descarteslabs.geo import DLTile

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))   # our helpers: config, periods, grid, ...

from config import Config, load_config
from grid import CHIP_PX, chip_ids
from lock import FolderLock, Locked
from periods import Period
from provenance import environment, repo_path, sha256

# AMW code. These must come after `grid`, which puts AMW's core/ on the search path.
import gee
from tf_darwin import apply_darwin_tf_compat
from tile_utils import cut_chips

SPECTRAL = list(gee.DataConfig._BAND_IDS["S2L1C"])   # AMW's 13 bands, model's order (gee.py:134)
EXTRA = ["n_obs", "n_clear", "clear_day_sum", "cs_sum", "cs_clear_sum"]   # ADDITION: see quality_bands
THIN_MAX = 2          # ADDITION: a pixel with 1 or 2 clear images is "thin" (thin_share)


# ------------------------------------------------------------ imagery

def quality_bands(s2, start_date: str, clear_threshold: float):
    """Five bands per pixel, from the Sentinel-2 images `s2` of the period."""
    # ADDITION: the images AMW's recipe could use (n_obs: those with a cloud score, cloudy
    # or clear), those passing its cloud test (n_clear: what the median is made of), the
    # sum of the clear images' days (clear_day_sum: day 0 is the period's first day), and
    # the sums of Cloud Score+ cs_cdf (1 clear, 0 cloudy) over all images (cs_sum) and over
    # the clear ones (cs_clear_sum). The sums become averages per chip (score_tile). Same
    # collections, dates and test as gee.py:208-229. A pixel with no image is masked and
    # arrives as 0; so does one with no clear image, in the bands of clear images.
    start = ee.Date(start_date)
    scores = s2.linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"),
                               ["cs_cdf"]).select("cs_cdf")
    clear = scores.map(lambda img: img.updateMask(img.gte(clear_threshold)))
    days = scores.map(lambda img: ee.Image.constant(img.date().difference(start, "day").floor())
                      .toFloat().updateMask(img.gte(clear_threshold)))
    return (scores.count().rename("n_obs").toUint16()
            .addBands(clear.count().rename("n_clear").toUint16())
            .addBands(days.sum().rename("clear_day_sum").toFloat())
            .addBands(scores.sum().rename("cs_sum").toFloat())
            .addBands(clear.sum().rename("cs_clear_sum").toFloat()))


class PanelExtractor(gee.GEE_Data_Extractor):
    """AMW's Earth Engine downloader, plus five bands that describe the images per pixel."""

    def _build_composite(self, start_date, end_date):
        # ADDITION: sent as 32-bit numbers. AMW's median arrives as 64-bit numbers (8 bytes
        # per band and pixel, about 40 MB a tile) and is cut to 32 bits on arrival
        # (gee.py:378, and score_tile scores 32-bit numbers), so cutting it on Earth
        # Engine's side instead gives the same values in half the download.
        composite = super()._build_composite(start_date, end_date).toFloat()   # AMW's median
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(start_date, end_date)
        return composite.addBands(quality_bands(s2, start_date, self.config.clear_threshold))

    def save_tile(self, pixels, tile, outdir, product_type="image"):
        """AMW's cache writer, made all or nothing (used only with --image-cache)."""
        # ADDITION: AMW writes the GeoTIFF in place (gee.py:469); a stop half-way would
        # leave a broken image that later runs read. Write it in a folder of its own,
        # then move it into the cache in one step.
        writing = Path(outdir) / f"_writing_{uuid.uuid4().hex}"
        writing.mkdir()
        try:
            path = super().save_tile(pixels, tile, writing, product_type)
            done = Path(outdir) / path.name
            os.replace(path, done)
        finally:
            shutil.rmtree(writing, ignore_errors=True)
        return done


def extractor_for(cfg: Config, period: Period, image_cache: Path | None) -> PanelExtractor:
    """AMW's downloader for one period, set up from the run's settings."""
    data_config = gee.DataConfig(
        tilesize=cfg.imagery.tilesize,
        pad=cfg.imagery.pad,
        collection=cfg.imagery.collection,
        bands=SPECTRAL + EXTRA,       # ADDITION: the extra bands come in the same request
        clear_threshold=cfg.imagery.clear_threshold,
        max_workers=cfg.imagery.max_workers,
        # DEVIATION from AMW: no image cache (AMW passes --image_cache_dir, pipeline.py:161).
        # Ours would hold roughly 200-300 TB, and AMW's stores whole numbers (gee.py:493),
        # so a rerun from it scores slightly different pixels. Every score here uses the
        # pixels exactly as downloaded, like AMW's first pass; resuming avoids re-downloads.
        # Tests may keep one (--image-cache) to score the same images again for free.
        image_cache_dir=image_cache,
    )
    # Dates as text, like AMW's driver. Earth Engine excludes the end date (Period.end).
    return PanelExtractor(period.start.isoformat(), period.end.isoformat(), data_config)


# ------------------------------------------------------------ scoring one tile

def score_tile(tile, pixels: np.ndarray, model, stride: int) -> pd.DataFrame:
    """One row per chip: AMW's confidence, the six votes behind it, quality columns."""
    spectral, extra = pixels[..., :len(SPECTRAL)], pixels[..., len(SPECTRAL):]
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

    return rows.assign(**quality(spectral, extra, bounds, stride))


def ratio(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """top / bottom as float32, missing where bottom is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(bottom > 0, top / bottom, np.nan).astype(np.float32)


def quality(spectral: np.ndarray, extra: np.ndarray, bounds, stride: int) -> dict:
    """Per chip: how well the period was seen, and what the composite looks like."""
    # ADDITION: per pixel first, then over each chip's square, the same squares as the
    # scores. Averages of a day or a cloud score are sums over the chip's pixels divided
    # by the images summed the same way, so a pixel with more images weighs more and an
    # empty one not at all. The three indices are averaged over the pixels with data;
    # "empty" = no clear image: 0 in every band, which the model sees as black.
    n_obs, n_clear, day_sum, cs_sum, cs_clear_sum = np.moveaxis(extra, -1, 0)
    band = {name: spectral[..., SPECTRAL.index(name)] for name in ("B2", "B4", "B8", "B10", "B11")}
    seen = ~(spectral == 0).all(axis=-1)

    def normalised(a, b):                       # (a - b) / (a + b), 0 where nothing was seen
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(seen & (a + b != 0), (a - b) / (a + b), 0)

    pixels = np.dstack([
        n_obs, n_clear, day_sum, cs_sum, cs_clear_sum,              # 0-4
        ~seen, (n_clear >= 1) & (n_clear <= THIN_MAX), seen,        # 5-7
        normalised(band["B8"], band["B4"]),                         # 8 NDVI, greenness
        normalised(band["B11"] + band["B4"], band["B8"] + band["B2"]),   # 9 bare soil (BSI)
        np.where(seen, band["B10"] / 10000.0, 0),                   # 10 cirrus reflectance, haze
    ]).astype(np.float32)
    layers, _ = cut_chips(pixels, bounds, CHIP_PX, stride, crs="epsg:4326")
    m = layers.mean(axis=(1, 2))                                    # (chips, 11)
    return {
        "n_obs": m[:, 0],
        "n_clear": m[:, 1],
        "n_clear_median": np.median(layers[..., 1].reshape(len(layers), -1), axis=1),
        "empty_share": m[:, 5],
        "thin_share": m[:, 6],
        "clear_day_mean": ratio(m[:, 2], m[:, 1]),
        "cs_mean": ratio(m[:, 3], m[:, 0]),
        "cs_clear_mean": ratio(m[:, 4], m[:, 1]),
        "ndvi_mean": ratio(m[:, 8], m[:, 7]),
        "bsi_mean": ratio(m[:, 9], m[:, 7]),
        "cirrus_mean": ratio(m[:, 10], m[:, 7]),
    }


# ------------------------------------------------------------ downloading and scoring

FLUSH_MINUTES = 2.0       # --flush-minutes default: save a part at least this often
SLOW_DOWNLOAD_S = 120     # a download slower than this gets its own line in _log.txt
REQUEST_TIMEOUT_S = 300   # give up on an Earth Engine answer silent for this long


def download(extractor, tile):
    """AMW's download of one tile (gee.py:289), with the moments it started and ended."""
    started = time.monotonic()
    pixels = extractor.get_tile_data(tile)
    return pixels, started, time.monotonic()


def stream_tiles(extractor, tiles: list, model, stride: int, running: dict):
    """Download tiles in parallel threads without pause; score each here as it arrives.

    Yields one dict per tile as soon as it is scored or has failed, and None after each
    second with nothing new, so the caller can save and report on time. `running`
    (download under way -> its tile and when it was asked for) is the caller's, to read.
    """
    # DEVIATION from AMW, in structure only: AMW starts the 500 downloads of a batch,
    # waits for the last one, scores, then starts the next batch (inference_engine.py:
    # 1167-1211), so the workers stand idle at the end of every batch while its slowest
    # downloads finish. Here a new download starts the moment one ends, over the whole
    # period, and saving is separate (run_period). At most two downloads per worker are
    # under way or waiting, so memory stays small. Same downloads, same scores.
    workers = extractor.config.max_workers
    waiting = iter(tiles)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            for tile in itertools.islice(waiting, 2 * workers - len(running)):
                running[pool.submit(download, extractor, tile)] = (tile, time.monotonic())
            if not running:
                return
            finished, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            if not finished:
                yield None
            for future in finished:
                tile, asked = running.pop(future)
                done = {"tile": tile, "rows": None, "error": None, "score_s": 0.0,
                        "download_s": time.monotonic() - asked}   # if it failed: since asked
                try:
                    pixels, started, ended = future.result()
                    done["download_s"] = ended - started
                    scoring = time.monotonic()
                    done["rows"] = score_tile(tile, pixels, model, stride)
                    done["score_s"] = time.monotonic() - scoring
                except Exception as err:         # as AMW: note it, retry it in the next try
                    done["error"] = f"{type(err).__name__}: {err}"
                yield done
    finally:
        # After the last tile, an error or a stop: drop the downloads still waiting and
        # don't wait for the running ones, so the folder's lock is given back at once;
        # the threads only download, and nothing they fetch after this is used.
        pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------ diagnostics

class RetryCounter(logging.Handler):
    """Counts the retries Google's libraries make inside AMW's download."""

    # ADDITION: AMW's download retries for up to 240 s (google.api_core, gee.py:288), and
    # the Earth Engine client retries "too many requests" answers itself (googleapiclient).
    # Both only log, and nobody reads those logs, so a throttled run just looks slow.
    # This handler catches their log lines; the count goes into every progress line.
    LOGGERS = {"google.api_core.retry": logging.DEBUG,     # "Retrying due to ..."
               "googleapiclient.http": logging.WARNING}    # "Sleeping ... before retry"

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.count, self.last = 0, ""

    def emit(self, record):
        self.count += 1
        self.last = record.getMessage()[:300]

    def attach(self) -> RetryCounter:
        for name, level in self.LOGGERS.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(self)
        logging.getLogger("google.api_core.retry").propagate = False   # no screen spam
        return self


RETRIES = RetryCounter()


class Log:
    """Progress lines on screen, and the same lines plus details in the folder's _log.txt."""

    # ADDITION: AMW logs to its tmux window only. A file in the folder ("_": Parquet
    # readers skip it) keeps the run's history next to its results, also on a cluster.
    def __init__(self, path: Path):
        self.path = path

    def write(self, line: str, screen: bool = True) -> None:
        if screen:
            try:
                print(line, flush=True)
            except OSError:              # the terminal was closed: keep going, log to file
                pass
        with open(self.path, "a") as f:
            f.write(f"{utc_now()}  {line.strip()}\n")


def describe_window(tiles: list[dict], seconds: float, workers: int, retries: int) -> str:
    """What happened since the last progress line, in one line."""
    ok = [t for t in tiles if t["error"] is None]
    if not tiles:
        return f"no tile finished in {seconds / 60:.1f} min, {retries} retries"
    download = np.array([t["download_s"] for t in ok]) if ok else np.zeros(1)
    busy = sum(t["download_s"] for t in tiles) / (workers * seconds)
    scoring = sum(t["score_s"] for t in ok) / seconds
    # "workers busy" near 100%: Earth Engine is the limit. Well below it while "scoring
    # busy" is near 100%: this computer's scoring is the limit, more workers won't help.
    return (f"{len(tiles) / seconds * 60:.0f} tiles/min, download median "
            f"{np.median(download):.0f} s (slowest {download.max():.0f} s), workers busy "
            f"{min(busy, 1):.0%}, scoring busy {min(scoring, 1):.0%}, {retries} retries")


# ------------------------------------------------------------ the period's folder

def tile_number(tile) -> int:
    """A tile's chip_ids without the last three digits (the chip), see grid.py."""
    return int(chip_ids(tile, 1)[0]) // 1000


def saved_tiles(folder: Path) -> tuple[set[int], int]:
    """The tiles already saved in this folder's part files, and how many chips."""
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        return set(), 0
    for p in parts:
        if "cs_mean" not in pq.read_schema(p).names:
            raise SystemExit(f"DETECT ERROR: {p} was written by detect.py before "
                             f"the quality columns of 23 Oct; this folder's parts would "
                             f"not match. Score into a new run, or move the folder aside")
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


MERGED = "part_00000.parquet"


def merge_parts(folder: Path) -> None:
    """Copy a complete period's parts into one file, part_00000.parquet, then remove them."""
    # ADDITION: a part every few minutes makes hundreds of small files per period, and a
    # cluster's file system (and every later read) is slow with many small files. Once
    # the period is complete, its parts become one file, read part by part so memory
    # stays small. If this stops half-way: before the rename, the parts are untouched
    # and _merge.tmp is removed next time; after it, part_00000 holds every row and the
    # next run_period removes the parts it copied.
    parts = sorted(folder.glob("part_*.parquet"))
    if len(parts) < 2:
        return
    tmp = folder / "_merge.tmp"
    with pq.ParquetWriter(tmp, pq.read_schema(parts[0]), compression="zstd") as out:
        for part in parts:
            out.write_table(pq.read_table(part))
    os.replace(tmp, folder / MERGED)
    for part in parts:
        if part.name != MERGED:
            part.unlink()


def write_status(folder: Path, about: dict, planned: list, errors: dict,
                 sessions: list, counted: tuple[set[int], int] | None = None) -> dict:
    """Count what the part files hold, and write it to _status.json, all or nothing.

    `counted` (saved tiles, chips) skips the recount, for the updates after each batch.
    """
    # ADDITION: AMW only logs how many tiles failed (inference_engine.py:1241). We record
    # which ones and why, and which code, packages and machine scored the period.
    saved, n_chips = counted if counted is not None else saved_tiles(folder)
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
               limit: int | None, model, members: list[str], folder: Path,
               lock: FolderLock, image_cache: Path | None, flush_minutes: float,
               tf_eager: bool) -> dict:
    """Score a period's tiles, saving a part every few minutes, with AMW's retries.

    The caller holds the folder's lock; it is checked before every write.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for leftover in folder.glob("_*.tmp"):
        leftover.unlink()                        # half-written by a crash; never valid
    if (folder / MERGED).exists():               # a merge stopped before removing the
        for part in folder.glob("part_*.parquet"):   # parts it copied (merge_parts)
            if part.name != MERGED:
                part.unlink()
    old = (json.loads((folder / "_status.json").read_text())
           if (folder / "_status.json").exists() else {})
    errors = old.get("tiles_failed", {})         # tile key -> its latest error
    sessions = old.get("sessions", []) + [
        {"started_utc": utc_now(), "finished_utc": None, "limit": limit,
         "max_workers": cfg.imagery.max_workers,       # ADDITION: this session's run-time
         "batch_tiles": cfg.inference.batch_tiles,     # settings, which --workers,
         "flush_minutes": flush_minutes,               # --batch-tiles, --flush-minutes
         "tf_eager": tf_eager,                         # --mac-eager
         "tries": cfg.inference.tries,                 # and --tries can change
         "image_cache": str(image_cache) if image_cache else None}   # kept images score
                                                       # slightly differently, so say so
        | environment(cfg)]
    about = {
        "period": period.tag,
        "frequency": period.frequency,
        "subregion": subregion,
        "start_date": period.start.isoformat(),
        "end_date_excluded": period.end.isoformat(),
        "ee_project": cfg.ee_project,
        "model": repo_path(cfg.inference.model_path),
        "model_members": members,                # vote_1, vote_2, ... in this order
    }
    write_status(folder, about, planned, errors, sessions)   # so even a crash leaves a record

    work = planned[:limit]                       # --limit: the first tiles only
    extractor = extractor_for(cfg, period, image_cache)
    stride = CHIP_PX // cfg.inference.stride_ratio           # AMW: inference_engine.py:756
    workers, size = cfg.imagery.max_workers, cfg.inference.batch_tiles
    saved, n_chips = saved_tiles(folder)
    log = Log(folder / "_log.txt")
    # ADDITION: say on screen what a resume skips, so a rerun visibly scores nothing twice.
    parts = len(list(folder.glob("part_*.parquet")))
    to_score = sum(tile_number(t) not in saved for t in work)
    log.write(f"  {period.tag}  " + (f"resuming: {len(saved):,}/{len(planned):,} tiles "
              f"already saved in {parts} part file(s), not scored again; " if saved
              else "starting: ") + f"{to_score:,} tiles to score, {workers} workers, "
              f"a part every {size} tiles or {flush_minutes:g} min")

    # Same as AMW: `tries` passes, each retrying the tiles the one before could not
    # finish (inference_engine.py:1151-1157). DEVIATION from AMW: a rerun resumes from
    # the part files; AMW starts over and relies on its image cache to be quick.
    for attempt in range(1, cfg.inference.tries + 1):
        todo = [t for t in work if tile_number(t) not in saved]
        if not todo:
            break
        if attempt > 1:
            log.write(f"  {period.tag}  try {attempt}: retrying {len(todo):,} tiles that "
                      f"failed in try {attempt - 1}")
        before = len(saved)
        scored, failed, window = [], {}, []      # since the last save / the last line
        last, retries = time.monotonic(), RETRIES.count
        running = {}                             # stream_tiles' downloads under way

        def save() -> None:
            """Write what is scored as the next part, update _status.json, say how it goes."""
            nonlocal scored, failed, window, last, retries, n_chips, errors
            lock.check()                         # still ours? never write into a stolen folder
            if scored:
                write_part(folder, [d["rows"] for d in scored])
                n_chips += sum(len(d["rows"]) for d in scored)
                saved.update(tile_number(d["tile"]) for d in scored)
            errors |= failed
            write_status(folder, about, planned, errors, sessions, counted=(saved, n_chips))
            now = time.monotonic()
            left = len(todo) - (len(saved) - before)
            rate = len(window) / (now - last)
            busy = sum(future.running() for future in running)
            ready = sum(future.done() for future in running)   # downloaded, not scored yet
            oldest = max((now - asked for _, asked in running.values()), default=0)
            hours = left / rate / 3600 if rate else None
            log.write(f"  {period.tag}  try {attempt}: {len(saved):,}/{len(planned):,} tiles "
                      f"saved (+{len(scored)}, {len(failed)} failed) | "
                      + describe_window(window, now - last, workers, RETRIES.count - retries)
                      + f" | {busy} downloading, {len(running) - busy - ready} waiting "
                        f"for a worker, {ready} waiting to be scored, oldest asked for "
                        f"{oldest:.0f} s ago"
                      + ("" if hours is None else f" | about {hours:.1f} h left in this try"
                         if hours >= 1 else f" | about {hours * 60:.0f} min left in this try"))
            if RETRIES.count > retries:
                log.write(f"    latest retry: {RETRIES.last}", screen=False)
            scored, failed, window, last, retries = [], {}, [], now, RETRIES.count

        # ADDITION: AMW saves nothing until a whole batch is scored; a stop loses the
        # batch. Here downloads never pause, and what is scored is saved as soon as
        # there are `batch_tiles` tiles or `flush_minutes` have passed, whichever is
        # first, so a stop loses a few minutes of work at most, usually nothing.
        stream = stream_tiles(extractor, todo, model, stride, running)
        try:
            for done in stream:
                if done is not None:
                    window.append(done)
                    if done["error"] is None:
                        scored.append(done)
                    else:
                        failed[done["tile"].key] = done["error"]
                        log.write(f"    failed {done['tile'].key} after "
                                  f"{done['download_s']:.0f} s: {done['error'][:300]}",
                                  screen=False)
                    if done["download_s"] > SLOW_DOWNLOAD_S:
                        log.write(f"    slow {done['tile'].key}: download "
                                  f"{done['download_s']:.0f} s", screen=False)
                if (STOP.is_set() or len(scored) >= size
                        or time.monotonic() - last >= flush_minutes * 60):
                    save()
                if STOP.is_set():
                    break
        finally:
            stream.close()                       # its downloads are dropped (stream_tiles)
        if scored or failed or window:
            save()
        if STOP.is_set():
            sessions[-1]["stopped_utc"] = utc_now()
            write_status(folder, about, planned, errors, sessions, counted=(saved, n_chips))
            log.write(f"  {period.tag}  stopped; every scored tile is saved", screen=False)
            raise Stopped(STOP_MESSAGE)

    sessions[-1]["finished_utc"] = utc_now()
    lock.check()
    status = write_status(folder, about, planned, errors, sessions)
    if status["complete"]:
        merge_parts(folder)
    log.write(f"  {period.tag}  session finished: {status['tiles_saved']:,}/"
              f"{status['tiles_planned']:,} tiles saved, {len(status['tiles_failed'])} "
              f"failed", screen=False)
    return status


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


def with_overrides(cfg: Config, workers: int | None, batch_tiles: int | None,
                   tries: int | None) -> Config:
    """The run's settings, with this command's run-time choices in place of the frozen ones."""
    # ADDITION: these three decide how fast a period is scored and in what pieces, never
    # what a score is, so a task may choose its own (a laptop and a cluster node differ).
    # Config is frozen, so `replace` makes a changed copy instead of editing it.
    imagery = cfg.imagery if workers is None else replace(cfg.imagery, max_workers=workers)
    inference = replace(cfg.inference,
                        batch_tiles=cfg.inference.batch_tiles if batch_tiles is None else batch_tiles,
                        tries=cfg.inference.tries if tries is None else tries)
    return replace(cfg, imagery=imagery, inference=inference)


STOP_SIGNALS = [getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGINT")
                if hasattr(signal, name)]   # time limit, closed terminal, Ctrl-C (no SIGHUP
                                            # on Windows)


class Stopped(SystemExit):
    """A stop request (Ctrl-C, a time limit, a closed terminal), as a normal Python exit."""


STOP = threading.Event()                 # set by the first stop request
STOP_MESSAGE = ("DETECT STOPPED; every scored tile is saved. Run the same command again "
                "to resume.")


def stop_on_signal(signum, frame) -> None:
    """First stop request: save what is scored, give the lock back, exit. Second: exit now."""
    # ADDITION: without this, a scheduler's time limit (SIGTERM) or a closed terminal
    # (SIGHUP) ends Python at once and leaves the lock behind until it goes stale. The
    # first request only raises a flag, which run_period checks after every tile, so a
    # stop never interrupts a write and loses only the downloads under way.
    name = signal.Signals(signum).name
    if not STOP.is_set():
        STOP.set()
        try:                             # os.write: print is not safe inside a handler
            os.write(2, f"\n{name}: saving what is scored, then stopping. Press Ctrl-C "
                        f"again to stop at once and lose it.\n".encode())
        except OSError:
            pass
        return
    for stop in STOP_SIGNALS:           # a third Ctrl-C must not interrupt the cleanup
        signal.signal(stop, signal.SIG_IGN)
    raise Stopped(f"DETECT STOPPED at once by a second {name}; tiles scored since the "
                  f"last part are lost. Run the same command again to resume.")


def check_image_cache(cfg: Config, folder: Path) -> Path:
    """The cache folder, after checking it holds images made with this run's settings."""
    # AMW's cache file names hold only the collection, tile and dates (gee.py:301), so
    # images made with other bands or another cloud test would be read without notice.
    # The folder remembers the settings of its first user, and refuses any other.
    folder = folder.expanduser().resolve()
    settings = {"collection": cfg.imagery.collection, "bands": SPECTRAL + EXTRA,
                "clear_threshold": cfg.imagery.clear_threshold,
                "tilesize": cfg.imagery.tilesize, "pad": cfg.imagery.pad}
    record = folder / "_settings.json"
    folder.mkdir(parents=True, exist_ok=True)
    if not record.exists():
        record.write_text(json.dumps(settings, indent=2))
    elif json.loads(record.read_text()) != settings:
        raise SystemExit(f"DETECT ERROR: {folder} holds images made with other settings "
                         f"({record.name}); use another --image-cache folder")
    return folder


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
    ap.add_argument("--workers", type=int,
                    help="parallel downloads (default: the run's imagery.max_workers)")
    ap.add_argument("--batch-tiles", type=int,
                    help="most tiles per part file (default: the run's inference.batch_tiles)")
    ap.add_argument("--flush-minutes", type=float, default=FLUSH_MINUTES,
                    help=f"save a part at least this often (default: {FLUSH_MINUTES:g})")
    ap.add_argument("--tries", type=int,
                    help="passes over failed tiles (default: the run's inference.tries)")
    ap.add_argument("--mac-eager", action="store_true",
                    help="on macOS, score with AMW's TensorFlow setting (tf_darwin.py): "
                         "about 1.6 times slower, use it if scoring crashes without it")
    ap.add_argument("--image-cache", type=Path,
                    help="tests only: keep each downloaded image in this folder and read it "
                         "from there next time (about 9 MB per tile and period; scores "
                         "from a kept image differ slightly from a fresh download)")
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
    for option in ("limit", "workers", "batch_tiles", "tries"):
        value = getattr(args, option)
        if value is not None and value < 1:
            raise SystemExit(f"DETECT ERROR: --{option.replace('_', '-')} must be at least 1")
    if not args.flush_minutes > 0:
        raise SystemExit("DETECT ERROR: --flush-minutes must be more than 0")
    cfg = with_overrides(cfg, args.workers, args.batch_tiles, args.tries)
    image_cache = check_image_cache(cfg, args.image_cache) if args.image_cache else None
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

    gee.EE_PROJECT = cfg.ee_project              # read by AMW's Earth Engine login (gee.py:73)
    gee._ensure_earth_engine_initialized()       # AMW's login (gee.py:47), first, so a login
                                                 # problem stops us before any folder is made
    # ADDITION: AMW sets no time limit on Earth Engine requests, and its client sets none
    # by default (ee/_state.py: deadline_ms = 0), so a request whose connection died (a
    # computer that slept, a network drop) waits forever and holds its worker: on
    # 2026-10-20 all 24 workers hung this way after a Mac went to sleep. With a limit, the
    # client retries such a request (up to 5 times, googleapiclient), then the tile fails
    # and is tried again in the next try. 300 s: the slowest download seen took 202 s.
    ee.data.setDeadline(REQUEST_TIMEOUT_S * 1000)
    # DEVIATION from AMW: AMW switches TensorFlow to step-by-step ("eager") execution on
    # macOS (tf_darwin.py, inference_engine.py:48), against crashes it saw with Apple's
    # graphics chips. By default we don't, so a Mac scores like Linux (AMW's VMs, the
    # cluster): compiled, 1.6 times faster on Ruby's Mac (bench_scoring.py, 2026-10-20),
    # scores within 1.2e-07 of eager ones (float32 rounding). --mac-eager restores it.
    tf_eager = args.mac_eager and platform.system() == "Darwin"
    if tf_eager:
        apply_darwin_tf_compat()
    model = tf.keras.models.load_model(str(cfg.inference.model_path),
                                       compile=False)          # AMW: inference_engine.py:424
    members = member_names(model)

    for stop in STOP_SIGNALS:
        signal.signal(stop, stop_on_signal)
    RETRIES.attach()

    print(f"{len(planned):,} tiles in subregion {args.subregion}, {len(periods)} "
          f"{args.frequency} periods, {periods[0].tag} to {periods[-1].tag}, "
          f"{cfg.imagery.max_workers} download workers, a part every "
          f"{cfg.inference.batch_tiles} tiles or {args.flush_minutes:g} min" + (f", first {args.limit} tile(s) only" if args.limit else "")
          + (f", images kept in {image_cache}" if image_cache else ""), flush=True)
    failed_periods, locked = 0, []
    try:
        for period in periods:
            if STOP.is_set():
                raise Stopped(STOP_MESSAGE)
            folder = (run_dir / "detect" / args.frequency / f"period={period.tag}"
                      / f"subregion={args.subregion}")
            try:
                with FolderLock(folder) as lock:     # given back when the block ends, however
                    status = run_period(cfg, period, args.subregion, planned, args.limit,
                                        model, members, folder, lock, image_cache,
                                        args.flush_minutes, tf_eager)
            except Locked as busy:                   # another process is writing this period
                print(f"{period.tag}  SKIPPED: {busy}", flush=True)
                locked.append(period.tag)
                continue
            print(f"{period.tag}  {status['tiles_saved']:,}/{status['tiles_planned']:,} "
                  f"tiles saved, {len(status['tiles_failed'])} failed  ->  {folder}",
                  flush=True)
            failed_periods += bool(status["tiles_failed"])
    except Stopped as stopped:
        # The lock is given back and every saved part is complete. Downloads may still be
        # running in their threads, and a normal exit would wait for them (up to AMW's
        # 240 s of retries, gee.py:288); nothing they fetch is used, so leave now.
        print(stopped.code, file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(1)
    problems = []
    if failed_periods:
        problems.append(f"{failed_periods} period(s) have failed tiles (see _status.json); "
                        f"run the same command again to retry them")
    if locked:
        problems.append(f"{len(locked)} period(s) skipped because another process is "
                        f"writing them: {', '.join(locked)}")
    if problems:
        raise SystemExit("\n".join(problems))
