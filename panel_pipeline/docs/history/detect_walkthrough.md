> **History, not maintained.** 15 Oct, written for a draft of detect.py that predates the lock, overrides and streaming. Read the real `../../detect.py`; this walkthrough is still useful for the imagery and scoring blocks (A–C).

# detect.py, block by block

plan.py fixed where (tiles and chips) and when (periods). detect.py fills in what: for every chip and period, AMW's score plus a few image counts. It is the only step that talks to Earth Engine, so it is the expensive one. Anything it doesn't save now would need a new download later.

It reuses AMW's code wherever it can:
- AMW's downloader (gee.py), through a subclass;
- AMW's chip cutter (tile_utils.py);
- AMW's four scoring lines, copied exactly.

The tests at the end show the scores are bit-identical to AMW's own inference code.

**Before you type it**
1. Pull my small edits with `git pull origin claude/project-thread-ls8p9r`. They:
   - add `provenance.py`, which holds plan.py's provenance helpers, moved unchanged so detect.py can use them too;
   - add an `earth_engine:` section to config.yaml (checked by config.py), whose `project:` is already set to `deforestation-and-mining`.

   plan.py now imports the helpers, and provenance.json keeps its exact format.
2. Run `python panel_pipeline/config.py`. It should print `ee project   deforestation-and-mining`. If your Cloud console shows a different project ID, put that ID in `earth_engine.project` instead.
3. Delete `tests_output/pilot` and plan it again with `python panel_pipeline/plan.py`, because the run's config copy needs the new setting.

The complete tested file is [detect.py](/mnt/project-files/notes/drafts/detect.py), to compare against yours.

## Block A: what it does, and imports

```python
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
```
- **One command covers one frequency and one subregion**, looping over periods. That is the unit a cluster job will take later. AMW also runs one command per subregion and period (pipeline.py:156-167).
- **The output folders are named like `period=2024-07/subregion=6`.** This is a standard convention called Hive partitioning: pandas, polars, DuckDB and R's arrow read the whole folder tree as one table and turn the folder names into columns. So `pd.read_parquet("tests_output/pilot/detect/monthly")` gives you every month at once, with `period` and `subregion` columns.
- pandas skips files whose names start with `_`, which is why the status file is `_status.json`. The only cost is that `=` looks odd in a folder name. Every filesystem and shell accepts it.
- **Import order matters here.** Importing `grid` runs `sys.path.append(.../core)`, and that is what makes `import gee` findable. So AMW's modules come last. An automatic import sorter could move them up, which is why the comment says it. Importing gee makes no Earth Engine call; AMW logs in only when asked.
- `SPECTRAL` reads AMW's own band table instead of retyping 13 names, so a typo or a swapped B8A/B8 can't creep in. The leading underscore in `_BAND_IDS` marks it as internal to AMW. We use it knowingly: the alternative, reading `.bands` off a DataConfig, gives a value a type checker sees as "might be None".

## Block B: the imagery: AMW's downloader plus two count bands

```python
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
```
- **Inheritance.** `class PanelExtractor(gee.GEE_Data_Extractor)` makes a new class that has every method of AMW's downloader: the Earth Engine login, the retry decorator, the `computePixels` request, and the reshaping into a (576, 576, bands) array. We replace exactly one method, `_build_composite`.
- Inside it, `super()._build_composite(...)` runs AMW's original. So the median composite is literally AMW's code, and we add two bands on top.
- **How AMW's code ends up calling ours.** AMW's `__init__` runs `self.composite = self._build_composite(...)`. Because `self` is a PanelExtractor, Python looks on our class first and finds our version. This is called overriding.
- **Why subclass rather than copy gee.py.**
  - For: a copy would freeze AMW's download code as of today, and a reader would have to compare 100 lines to see what changed. With the subclass, our change is these 12 lines.
  - Against: we depend on the name of an internal AMW method. If AMW renamed it, our version would no longer be called. Every request would then ask for two bands that don't exist and fail, so it would break loudly, not silently.
- **The two counts:**
  - `scores` is the period's images, cut down to their cloud-score band. `.count()` gives, per pixel, how many images have a score at all: `n_obs`.
  - `clear` masks the cloudy pixels with AMW's own test (cs_cdf ≥ 0.6). Its `.count()` is the number of clear images the median is made of: `n_clear`.
  - Where two Sentinel-2 tiles overlap, one satellite pass is in the collection twice, so it counts twice. That's also how it enters AMW's median. Worth knowing for the paper.
- **Earth Engine is lazy.** These lines only describe a computation. Nothing runs, and nothing is billed, until a tile is requested. That's why making an extractor is instant.
- `bands=SPECTRAL + COUNTS`: AMW's `get_tile_data` requests exactly `config.bands`. So the 13 bands and the 2 counts arrive in one request, on the same pixel grid. That is a little more compute per request, and no extra requests.
- **No image cache:** see the DEVIATION comment, and choice 3 in my earlier message.

## Block C: scoring one tile

```python
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
```
- `pixels[..., :13]`: `...` means "every axis before the last", so this takes the first 13 bands of the (576, 576, 15) array. It's a view, not a copy.
- **The four AMW lines are copied exactly:**
  1. divide by 10,000 and clip to 0..1;
  2. cut 529 chips of 48 × 48 pixels with a stride of 24, so neighbouring chips overlap by half;
  3. predict;
  4. average the six votes.

  `votes` has shape (529, 6): one row per chip, one column per ensemble member.
- **Why `confidence` is stored as float64 (8 bytes), not float32 (4 bytes).** AMW averages in float32, then widens to float64. If we stored float32, pandas would compare `confidence > 0.4` in float32, which turns 0.4 into 0.4000000059604645. A chip scoring exactly that would pass AMW's test but fail ours. I checked it: `np.float32(0.4) > 0.4` is False, while AMW's float64 version is True. It's one value among millions, but with 20 billion rows I'd expect it a few dozen times around each threshold.
- `rows[f"vote_{i + 1}"] = votes[:, i]` adds one column per member. Which member is which is recorded in `_status.json` (Block G, `member_names`).
- **The quality columns:**
  - `(spectral == 0).all(axis=-1)` is True where all 13 bands are 0: no clear image, so the model sees a black pixel there.
  - `np.dstack` stacks the two count layers and this empty mask into one (576, 576, 3) array, and AMW's `cut_chips` cuts it into the same 529 squares.
  - `.mean(axis=(1, 2))` averages each chip over its 48 × 48 pixels. The result is a (529, 3) array: mean n_obs, mean n_clear, and the share of empty pixels. The mean of a True/False mask is the share of True values.

## Block D: one batch of tiles, downloaded in parallel

```python
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
```
- **Threads.** A download is mostly waiting on the network, so several can wait at once. Python runs other threads while one waits on the network. TensorFlow also lets go while it computes, so downloads continue while this thread scores.
- `ThreadPoolExecutor` is a pool of worker threads. `pool.submit(f, tile)` hands it a job and immediately returns a **Future**: a receipt you later exchange for the result. `wait(..., return_when=FIRST_COMPLETED)` sleeps until at least one download is done.
- `itertools.islice(waiting, n)` takes the next n tiles from the iterator `waiting`, or fewer at the end. So at most 2 × workers downloads are under way, or finished but not yet scored. The workers never run dry, and memory stays around 16 tiles (0.3 GB).
- **Why this differs from AMW (structure only).** AMW submits all 500 downloads of a batch at once and keeps every result until the batch ends: about 9 GB. On a GPU that hardly matters. On a laptop or a CPU node, scoring is slower than downloading, so the results pile up. The downloads and scores are the same either way.
- `future.result()` returns the pixels, or raises the download's error here. So one `try` catches both download errors and scoring errors.
- `except Exception` deliberately lets Ctrl-C through (that's a KeyboardInterrupt, not an Exception), so you can still stop the program.
- A failed tile is only noted, with its error text, as in AMW. It gets another chance in the next try (Block F).

## Block E: the period's folder: part files and the status record

```python
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
```
- `tile_number` is a tile's chip_ids without the last three digits. Every chip_id contains its tile, so the part files themselves say which tiles are done. There is no separate list that could disagree with them.
- `saved_tiles` reads only the chip_id column of each part. Parquet stores columns separately, so it skips the other ten. It also refuses if a chip appears twice, which can only happen if two copies of detect.py wrote into the same folder.
- **All or nothing.** `write_part` and `write_status` first write a temporary file, then `os.replace` it to the final name. A rename within one disk is atomic: the file is either the old one or the complete new one, never half of each. So a crash can't leave a broken part that looks finished. The temporary names start with `_`, so readers skip them too.
- `max(numbers, default=0) + 1` is the next part number; `default=0` covers an empty folder.
- Each part is sorted by chip_id, so its content doesn't depend on which download happened to finish first.
- `compression="zstd"` gives smaller files than pandas' default (snappy), for a little more CPU when writing. Any Parquet reader opens it.
- `write_status` counts from the files, not from memory:
  - **missing:** planned tiles that are in no part;
  - **failed:** missing tiles that have an error;
  - **not tried:** the rest, for example tiles beyond `--limit`.

  `about | {...}` merges two dicts into one (Python 3.9 and later).
- `utc_now` writes times in UTC with the offset spelled out, so runs on machines in different time zones compare correctly.

## Block F: one period

```python
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
        "ee_project": cfg.ee_project,
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
```
- **Start of a session:**
  1. Delete half-written temporaries a crash may have left.
  2. Read the old `_status.json`, to keep earlier errors and sessions.
  3. Add this session: start time, machine, code version and package versions (`environment`, from provenance.py).
  4. Write the status straight away, so even a session that crashes leaves a record of which code ran.
- `planned[:limit]`: a slice up to `None` means "to the end", so without `--limit` this is every tile.
- There is one extractor per period, because its composite is that period's median.
- **The tries loop is AMW's.** Pass 1 tries every unsaved tile, pass 2 the ones that failed, pass 3 those again (`tries: 3`, as AMW's driver passes, pipeline.py:161).
- Within a pass, tiles go in batches of 500 (AMW's batch size). Each batch becomes one part file, so a crash loses at most one batch.
- **Where it differs from AMW:** a rerun of AMW starts over, and is quick only because of its cache. A rerun here skips every saved tile, because `saved` starts from the part files.
- `saved |= {...}` adds to a set in place. `errors |= failed` merges dicts in place, so a tile's latest error replaces an older one.

## Block G: checks before scoring

```python
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
```
- `check_model` recomputes the model's fingerprint and compares it with the one plan.py recorded. One run uses one model, so a swapped model file can't slip into the middle of a run.
- `pick_periods`: `Period` objects compare in time order. The dataclass has `order=True`, which compares year, then month, then length. The frequency check stops a quarterly tag such as `--from 2018Q1` in a monthly run, which would otherwise compare oddly.
- `subregion_tiles` takes the tiles.csv rows for this subregion, in AMW's order. `DLTile.from_key` rebuilds a tile from its key alone. That's pure arithmetic, so no network is needed. I checked that all 80 pilot tiles rebuilt this way give the planned chip squares, bit for bit.
- `member_names`: the model's last layer is a Concatenate that sets the six members' outputs side by side. Its inputs, in order, are the members. So vote_1 is 48px_v4.10-CNN800kb, and so on to vote_6, 48px_v4.22-CNN800kc.

## Block H: the command line

```python
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

    gee.EE_PROJECT = cfg.ee_project              # read by AMW's Earth Engine login (gee.py:73)
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
```
- **The checks run cheapest first**, so a typo fails within a second, before the model loads or Earth Engine is contacted.
- `--from` is a Python keyword, so its value is stored as `args.first` (`dest="first"`).
- **No separate Andes run:** see the DEVIATION comment. We already keep every score of every tile, so the Andes chips' scores are in the subregion runs.
- **The Earth Engine project, then the login, first.** AMW's login reads `gee.EE_PROJECT`, so we set it from `earth_engine.project` in the run's config. AMW logs in when its first downloader is created. We log in right away instead, so a login problem stops everything before any folder is made. Without a login you'll see `Please authorize access to your Earth Engine account by running earthengine authenticate`.
- `apply_darwin_tf_compat()` is AMW's macOS setting for TensorFlow, which AMW applies whenever its inference code is imported. On Linux it does nothing.
- The model is loaded once and reused for every period.
- **The exit code** is 1 if any period has failed tiles. You won't notice it by hand, but a cluster scheduler can, and rerun the job.

## Testing it

**What I tested here.** My container has no Earth Engine, so these use fake imagery:
1. **AMW's own `get_tile_data`**, with Earth Engine's reply mocked and its bands shuffled. Our 15 bands came out in the requested order, with exact values.
2. **`score_tile` against AMW's own `InferenceEngine.predict_on_tile_pixels`**, on 6 tiles (3,174 chips). Same chips, bit-identical confidence, identical votes.
3. **The quality columns**, recomputed by hand for chips 0, 1 and 23. This also confirms AMW's chip order: down a column first, so chip 1 is below chip 0, and chip 23 starts the second column.
4. **End to end, on an 8-tile test area:**
   - A tile that always fails: recorded with its error, and the exit code is 1.
   - Tiles that fail once: retried and saved.
   - A crash in the middle of a month: the rerun fetched only the unsaved tiles.
   - `--limit`, and the quarterly dates (2018-01-01 to 2018-04-01).
   - Every refusal, and the duplicate guard.
   - Reading the whole tree with one `pd.read_parquet`.

**What only your first real test can check:** the Earth Engine part itself, i.e. that the request works with the two count bands.

**Your first test** makes one Earth Engine request. Do it after the three steps at the top, and after `earthengine authenticate` if you haven't logged in on this machine:
```
python panel_pipeline/detect.py --run tests_output/pilot --frequency monthly --subregion 6 --from 2024-07 --to 2024-07 --limit 1
```
July is dry season in Madre de Dios, so the tile should be mostly clear. Then look at the numbers:
```
python -c "import pandas as pd; d = pd.read_parquet('tests_output/pilot/detect/monthly'); print(len(d)); print(d.describe().T[['mean', 'min', 'max']])"
```
What to expect:
- 529 rows.
- `n_obs` around 6: in 2024, two satellites passed every 5 days. It can reach about 12 where two Sentinel-2 tiles overlap.
- `n_clear` no higher than `n_obs`.
- `empty_share` near 0 in July.
- `confidence` between 0 and 1.

Then score the full month, 80 tiles: run the same command without `--limit`. It takes a few minutes: the model needed about 2.3 seconds per tile on my 4-core test machine. Running it a second time should finish at once, because every tile is already saved.

## Deviations in this file

`grep -n "DEVIATION\|ADDITION" panel_pipeline/detect.py` lists them:

| Where | Marker | What | Why |
|---|---|---|---|
| PanelExtractor | ADDITION | n_obs and n_clear bands | quality controls, only computable while downloading |
| extractor_for | DEVIATION | no image cache | about 200 to 300 TB, and AMW's cache stores rounded pixels |
| score_tile | DEVIATION | every chip kept; confidence as float64 | thresholds applied later; exact AMW comparisons |
| score_tile | ADDITION | n_obs, n_clear, empty_share per chip | quality controls for the panel |
| score_batch | DEVIATION, structure only | at most 2 downloads per worker waiting | memory (AMW holds about 9 GB per batch) |
| write_status | ADDITION | failed tiles, their errors, code version per session | AMW only logs a count |
| run_period | DEVIATION | a rerun resumes from the part files | AMW starts over and relies on its cache |
| main | DEVIATION | no separate Andes run | the Andes scores are already in the subregion runs |
