# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
panel.py — build a per-period CHIP PANEL: model probability + cloud/coverage
flags for EVERY 48-px chip in the AOI, one row per chip, one file per period.

WHY A PANEL AND NOT THE PIPELINE'S GEOJSON
------------------------------------------
The upstream pipeline (core/inference_pipeline.py -> InferenceEngine.bulk_predict)
only writes chips whose probability EXCEEDS the threshold, cumulatively appended
per region name. For econometrics we need the opposite of a filtered list:

  * ALL chip probabilities (a continuous outcome; thresholding is a modelling
    choice we want to vary in the analysis, not bake into the data), and
  * an explicit MISSINGNESS indicator per chip (clear_frac; see coverage.py),
    because a fully clouded chip scores ~0 and would otherwise masquerade as
    "no mining".

Output per period: a GeoParquet with one row per chip:
  chip_id, tile_key, row, col, prob, clear_frac, n_clear_mean, n_clear_min,
  n_total_mean, geometry (EPSG:4326), + period metadata columns.

SCHEMA v2 (2026-08-04) ADDS
---------------------------
* ``p_m0 … p_m5`` — the RAW per-member probabilities. The model is a Concatenate of six
  single-sigmoid CNNs; ``_probs_from_predictions`` averages them and throws the members
  away (core/inference_engine.py:632). Their spread is the only uncertainty signal the
  model emits, costs nothing to keep, and is unrecoverable later without re-running all
  inference. ``prob`` remains exactly ``mean(p_m*)``. See ``member_probs``.
  Measured cost: ~+2.0 MB per period parquet (1.68 -> ~3.7 MB); zero extra compute.
* Provenance columns, constant per file, dictionary-encoded to a few hundred bytes:
  ``schema_version, model, model_sha256, ensemble_members, n_members,
  clear_threshold, collection, tilesize, pad, chip_size, stride, git_commit``.
  Rationale: ``analysis.py:94`` globs every ``panel_*.parquet`` in a directory with no
  filter, so panels produced under different models or cloud thresholds would otherwise
  pool silently. ``clear_threshold`` in particular is baked into the composite AND absent
  from the tile-cache key (gee.py:178) — it cannot be inferred from the data afterwards.
Plus a per-tile stats CSV (timing, coverage, fill-zero verification) and a
convenience GeoJSON of positives (prob > threshold) for eyeballing in QGIS.

DESIGN CONSTRAINTS
------------------
* NO upstream file is modified. We reuse the engine's own building blocks
  (get_tile_data, cut_chips, _model_infer, _probs_from_predictions) so the
  probabilities are bit-identical to what bulk_predict would produce.
  Pinned against upstream/main c473933; if a future sync breaks an attribute,
  the asserts below fail loudly rather than silently drifting.
* Chip windows are re-derived with the SAME iteration order as
  core/tile_utils.cut_chips (col-major outer loop) — asserted at runtime —
  so chip i's geometry, pixels, and coverage stats always refer to the same
  window.
* chip_id = f"{tile.key}:c{i}" is deterministic across periods (tiling and
  chipping depend only on the AOI), giving a balanced panel: same chip_id set
  in every period file.
"""
# >> FILE OVERVIEW: fork-original (Ruby's research layer; not in upstream Amazon Mining
# >> Watch).
# >> Builds the "panel": one row per 48 px chip per period (month or quarter), for EVERY
# >> chip, not only detections. Reuses upstream core/gee.py (imagery), core/tile_utils.py
# >> (tiles, chips) and core/inference_engine.py (CNN scores), and adds cloud counts from
# >> coverage.py.
# >> Why a panel: keep the continuous prob (threshold later, in analysis) plus a
# >> missingness flag (clear_frac), so a clouded chip is never mistaken for "no mining".
# >> Called by run_trials.py:main; the parquet files are read by analysis.py:load_panels.
from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# >> geopandas = pandas tables with a geometry column; writes GeoParquet and GeoJSON here.
import geopandas as gpd
import numpy as np
import pandas as pd

# >> gee = upstream Earth Engine download (DataConfig, tile cache); tile_utils = DLTile
# >> grid and cut_chips; inference_engine = model loading and scoring.
# upstream core/ modules (sys.path prepared by run_trials.py)
import gee
import tile_utils
from inference_engine import InferenceConfig, InferenceEngine

# >> Fork module: per-pixel N_CLEAR / N_TOTAL scene counts and per-chip summaries.
from coverage import CoverageExtractor, chip_coverage_stats

log = logging.getLogger(__name__)

# >> Repo root = two folders above this file; used by _git_commit for provenance.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Bump when the panel column set changes. Stored in every row so panels written by
#: different code versions can never be pooled silently by analysis.py (which globs
#: panel_*.parquet with no filter — analysis.py:94).
#:   1 = chip_id..geometry + period/cadence/start/end   (runs up to 2026-08-04)
#:   2 = + per-member probabilities p_m* + provenance columns
# >> Current schema is 2. analysis.py:_check_provenance warns if panels mix versions.
PANEL_SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# ensemble members and provenance                                             #
# --------------------------------------------------------------------------- #
# >> What: keeps each of the six CNN members' own score, not only their average.
# >> In: raw model output, float array (N, 6) for the production ensemble (N = 529 per
# >> tile).
# >> Out: the same array as float64, or None if it is not 2-D with >= 3 columns.
# >> Why: member disagreement is the model's only uncertainty signal; upstream averages it
# >> away in _probs_from_predictions. Feeds panel columns p_m0..p_m5.
# >> Called by build_period_panel; `prob` itself still comes from the upstream mean.
def member_probs(raw: np.ndarray) -> Optional[np.ndarray]:
    """Per-member positive probabilities, shape (n_chips, n_members), or None.

    WHY THIS EXISTS
    ---------------
    The production model is a Keras ``Concatenate`` of six ``Sequential`` CNNs, each
    ending in a single sigmoid unit — verified by reading the model_config of
    ``models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5``:

        48px_v4.10-CNN800kb  48px_v4.18-CNN800kd  48px_v4.20-CNN800kg
        48px_v4.21-CNN100ka  48px_v4.22-CNN800kb  48px_v4.22-CNN800kc

    So ``InferenceEngine._model_infer`` returns shape (n, 6), and
    ``_probs_from_predictions`` (core/inference_engine.py:632) collapses it with
    ``np.mean(axis=1)``. The spread across members is the only uncertainty signal
    this model emits, it costs nothing to keep (the array already exists in memory),
    and it is unrecoverable afterwards without re-running all inference. So we keep
    the raw members rather than a summary statistic: std/min/max are a modelling
    choice better made in analysis.py, where it can be revised.

    Returns None when there is nothing to disagree about: a single-head model
    (shape (n,) or (n,1)) or a 2-column softmax. The ``>= 3`` cutoff mirrors
    ``_probs_from_predictions``, which only takes a row mean for shape[1] > 2.
    """
    arr = np.asarray(raw)
    # >> Single-output model or 2-class softmax: nothing to keep.
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    # >> Keras returns float32, so this converts to float64 (copy=False skips a needless
    # >> copy).
    return arr.astype(np.float64, copy=False)


# >> What: records which code version produced the panel (git_commit column).
# >> Runs `git rev-parse --short HEAD` and `git status --porcelain --untracked-files=no`.
# >> Out: e.g. "a1b2c3d" or "a1b2c3d-dirty"; "unknown" if git fails. Never raises.
# >> Why untracked files are ignored: run outputs inside the repo would mark every run
# >> dirty.
def _git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Short commit hash, suffixed ``-dirty`` if TRACKED files have local modifications.

    ``--untracked-files=no`` is deliberate: a run writes its own outputs into the tree
    (panels, logs, run directories), so counting untracked files would mark every single
    panel ``-dirty`` and make the flag meaningless. What matters for provenance is
    whether the *committed code* differs from what is running.
    """
    try:
        # >> subprocess call 1: short hash of the checked-out commit (5 s timeout).
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return "unknown"
        # >> subprocess call 2: any modified TRACKED file -> "-dirty" suffix (15 s
        # >> timeout).
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=15)
        dirty = "-dirty" if status.stdout.strip() else ""
        return head.stdout.strip() + dirty
    except Exception:  # git absent, not a repo, timeout — never fail a run over this
        return "unknown"


# >> What: fingerprint of the model weights file, so panels from different weights never
# >> pool.
# >> Reads the file in 1 MiB blocks (1 << 20 bytes) and hashes it with SHA-256 (hashlib).
# >> Out: first 16 hex characters (64 bits); "unknown" on any error.
# >> Feeds the model_sha256 column; analysis.py:_check_provenance warns if it varies.
def _file_sha256(path: Path, nbytes: int = 16) -> str:
    """First ``nbytes`` hex chars of the file's SHA-256. Identifies the weights."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            # >> Read 1 MiB at a time until an empty read: constant memory for the ~16 MB
            # >> .h5 file.
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:nbytes]
    except Exception:
        return "unknown"


# >> What: lists the names of the six sub-models inside the ensemble.
# >> Walks engine.model.layers (Keras) and keeps the Sequential layers, in model order,
# >> which is also the Concatenate output order, i.e. the order of p_m0..p_m5.
# >> Out: comma-joined string, e.g. "48px_v4.10-CNN800kb,48px_v4.18-CNN800kd,..."; '' on
# >> error.
# >> Feeds the ensemble_members column.
def _member_names(engine) -> str:
    """Comma-joined names of the ensemble's member sub-models, or '' if unavailable."""
    try:
        names = [lyr.name for lyr in engine.model.layers
                 if lyr.__class__.__name__ == "Sequential"]
        return ",".join(names)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# chip-window bookkeeping                                                     #
# --------------------------------------------------------------------------- #
# >> What: lists the top-left pixel corner of every chip in a tile, in the pipeline's
# >> order.
# >> In: width, height in px (576, 576), chip_size 48, stride 24.
# >> Out: generator of (row, col) ints; 23 x 23 = 529 corners for a 576 px tile.
# >> Order: outer loop over columns, inner over rows (column-major), same as cut_chips.
# >> Why: lets build_period_panel zip() windows with probs by position, with no index
# >> join.
def chip_windows(width: int, height: int, chip_size: int, stride: int):
    """Yield (row, col) upper-left corners in EXACTLY the order of
    core/tile_utils.cut_chips (outer loop over col, inner over row).
    Keeping the order identical is what lets us zip() windows with the
    chips/geometries/probabilities arrays without an index join."""
    # >> Stop = 576 - 48 + 24 = 552, so corners are 0, 24, ..., 528 (last chip ends at px
    # >> 576).
    for col in range(0, width - chip_size + stride, stride):
        for row in range(0, height - chip_size + stride, stride):
            yield row, col


# >> What: checks that "cloudy in every scene" pixels really come back from EE as all
# >> zeros.
# >> In: pixels (576, 576, 13) raw reflectance x 10,000; cov (576, 576, 2) [N_CLEAR,
# >> N_TOTAL].
# >> Out: float in [0, 1] = share of pixels where the two "has data" masks disagree. 0.0
# >> means the all-zero rule is a valid cloud flag, even for older cached imagery.
# >> Covers the whole 576 px tile including the 12 px pad. Written to the tile_stats CSV
# >> (fill0_mismatch) by build_period_panel; summarised in analysis.py:section_fill_check.
def fill_zero_mismatch(pixels: np.ndarray, cov: np.ndarray) -> float:
    """Empirical check of the fill=0 assumption (pipeline card §5).

    valid_from_counts : N_CLEAR > 0  (>=1 clear scene entered the median)
    valid_from_image  : not all 13 bands exactly 0

    If Earth Engine fills fully-masked pixels with 0 (as assumed), the two masks
    agree everywhere and this returns 0.0. Persistent non-zero values would mean
    gaps are NOT reliably detectable from the imagery alone — the coverage
    raster would then be the only trustworthy flag. Reported per tile, averaged
    in the analysis report."""
    # >> Mask 1: at least one clear scene fed the median (from the coverage counts).
    valid_from_counts = cov[..., 0] > 0
    # >> Mask 2: not all 13 bands exactly 0 (from the imagery); axis=-1 is the band axis.
    valid_from_image = ~np.all(pixels == 0, axis=-1)
    # >> != compares the masks pixel by pixel; mean of booleans = fraction that disagree.
    return float(np.mean(valid_from_counts != valid_from_image))


# --------------------------------------------------------------------------- #
# per-period build                                                            #
# --------------------------------------------------------------------------- #
# >> What: describes one time window to composite, e.g. January 2025 or Q1 2025.
# >> Fields: label ('m2025-01' / 'q2025Q1', used in every output filename), start, end,
# >> cadence ('monthly' / 'quarterly'). end is EXCLUSIVE: 2025-01-01 -> 2025-02-01 covers
# >> all of January (ee filterDate rule).
# >> Built by run_trials.py:month_specs / quarter_specs; consumed by build_period_panel.
@dataclass
class PeriodSpec:
    """One composite window. label examples: 'm2025-01', 'q2025Q1'."""
    label: str
    start: str   # YYYY-MM-DD, inclusive
    end: str     # YYYY-MM-DD, EXCLUSIVE (ee.filterDate convention, as upstream)
    cadence: str  # 'monthly' | 'quarterly'


# >> What: scores every chip of the AOI for one period and saves it as one table (the
# >> panel).
# >> In: PeriodSpec, region GeoJSON, model .h5, output and cache folders. pred_threshold
# >> only decides the convenience positives GeoJSON, never which rows enter the panel.
# >> Out: path of panels/panel_<label>.parquet (GeoParquet, EPSG:4326, one row per chip;
# >> 42,320 rows for La Pampa = 80 tiles x 529 chips). Also writes
# >> ../tiles/tile_stats_<label>.csv and ../detections/detections_<label>.geojson.
# >> Flow: tile the AOI -> per batch, fetch imagery + coverage in parallel -> score
# >> serially.
# >> Called by run_trials.py:main. Gotcha: rows stay in RAM; a crash mid-period writes
# >> nothing.
def build_period_panel(
    spec: PeriodSpec,
    region_path: Path,
    model_path: Path,
    out_dir: Path,
    cache_dir: Path,
    pred_threshold: float = 0.4,
    max_workers: int = 8,
    limit_tiles: Optional[int] = None,
) -> Path:
    """Fetch imagery + coverage for every tile of the AOI in this window, score
    every chip, and write the period's panel parquet. Resume-safe at file level
    (caller skips existing panels) and cheap to re-run (tile cache).

    Returns the path of the written parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # >> Filename contract: panel_<label>.parquet; run_trials.py checks it to resume.
    panel_path = out_dir / f"panel_{spec.label}.parquet"

    # --- tiling: identical parameters to the production pipeline ------------
    # >> FILE READ: AOI polygon(s) from GeoJSON (lat/lon, EPSG:4326).
    region = gpd.read_file(region_path)
    # >> Merge all AOI polygons into one shape (union_all is the geopandas >= 1.0 name).
    try:
        geom = region.union_all()
    except AttributeError:  # geopandas < 1.0
        geom = region.unary_union
    # >> Upstream defaults: S2L1C, tilesize 552 + pad 12 = 576 px at 10 m, clear_threshold
    # >> 0.6.
    data_config = gee.DataConfig(image_cache_dir=str(cache_dir))
    # >> DLTile grid maths only (descarteslabs, no API call): list of ~5.5 km UTM squares.
    tiles = tile_utils.create_tiles(geom, data_config.tilesize, data_config.pad)
    # >> Smoke-test option: keep only the first N tiles.
    if limit_tiles:
        tiles = tiles[:limit_tiles]
    log.info("[%s] %d tiles, window %s..%s", spec.label, len(tiles),
             spec.start, spec.end)

    # --- engine (embedding_strategy='none' => from-scratch CNN path) --------
    # >> Upstream InferenceEngine on the production path: CNN on raw pixels, no foundation
    # >> model. start/end go straight into ee filterDate, so the end date stays exclusive.
    engine = InferenceEngine(
        start_date=spec.start, end_date=spec.end,
        data_config=data_config,
        config=InferenceConfig(model_path=Path(model_path),
                               embedding_strategy="none",
                               pred_threshold=pred_threshold),
    )
    # >> The private methods reused below; assert = stop with an error if any is missing.
    # fail loudly if a future upstream sync renames the internals we reuse
    for attr in ("_model_infer", "_probs_from_predictions", "_resolve_chip_params"):
        assert hasattr(engine, attr), (
            f"InferenceEngine.{attr} missing — upstream API changed; "
            "re-pin timeseries/panel.py against the new core/inference_engine.py")
    # >> Loads the Keras .h5 ensemble once: input (48, 48, 13), output (N, 6).
    engine._ensure_keras_model()
    # >> Upstream chip settings: chip_size 48 px (~480 m), stride 24 px (half overlap).
    chip_size, stride = engine._resolve_chip_params()

    # >> Coverage fetcher for the same window and SAME cloud threshold as the imagery.
    cov_extractor = CoverageExtractor(
        spec.start, spec.end,
        clear_threshold=data_config.clear_threshold,  # MUST match imagery pull
        cache_dir=cache_dir)

    # Provenance computed ONCE per period (the sha256 reads 16 MB, ~50 ms).
    # These columns are constant per file, so parquet dictionary-encodes them to a
    # couple of hundred bytes each — measured negligible against the 1.7 MB baseline.
    # >> Provenance columns, identical on every row of this period's file.
    prov = {
        "schema_version": PANEL_SCHEMA_VERSION,
        # The AOI defines the whole chip universe. Without it, panels built over different
        # regions pool with no warning — chip_ids simply don't overlap, which looks like an
        # unbalanced panel rather than an error.
        "region": Path(region_path).name,
        "model": Path(model_path).name,
        "model_sha256": _file_sha256(Path(model_path)),
        "ensemble_members": _member_names(engine),
        "clear_threshold": float(data_config.clear_threshold),
        "collection": data_config.collection,
        "tilesize": int(data_config.tilesize),
        "pad": int(data_config.pad),
        "chip_size": int(chip_size),
        "stride": int(stride),
        "git_commit": _git_commit(),
    }
    log.info("[%s] provenance: model=%s sha=%s clear_threshold=%.2f "
             "chip=%d stride=%d commit=%s", spec.label, prov["model"],
             prov["model_sha256"], prov["clear_threshold"], prov["chip_size"],
             prov["stride"], prov["git_commit"])

    # >> rows = one dict per chip (panel rows); tile_stats = one dict per tile (CSV rows).
    rows: List[dict] = []
    tile_stats: List[dict] = []

    # Process in batches: imagery + coverage fetched concurrently (network
    # bound), CNN inference serialized (TF session; also keeps RAM bounded to
    # ~max_workers tiles of 576x576x13 float32 ≈ 17 MB each).
    # >> Loop over tiles in batches of max_workers (8 by default; run_staged.sh passes 4).
    # >> Tile order = create_tiles order, so panel rows are ordered tile, then chip.
    for b0 in range(0, len(tiles), max_workers):
        batch = tiles[b0:b0 + max_workers]

        t0 = time.perf_counter()
        # >> Thread pool: 2 jobs per tile (imagery + coverage), up to max_workers threads.
        # >> Threads, not processes: the work is mostly waiting on Earth Engine HTTP
        # >> (network I/O).
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # >> Each submit = one ee.data.computePixels call, or a cache hit on disk.
            pix_futs = {t.key: ex.submit(_fetch_pixels_safe, engine, t)
                        for t in batch}
            cov_futs = {t.key: ex.submit(_fetch_cov_safe, cov_extractor, t)
                        for t in batch}
            # >> .result() waits for each download; the _safe wrappers return None instead
            # >> of raising.
            pixels_by_key = {k: f.result() for k, f in pix_futs.items()}
            cov_by_key = {k: f.result() for k, f in cov_futs.items()}
        # >> fetch_s = wall time for the whole batch; later divided by the batch size.
        fetch_s = time.perf_counter() - t0

        # >> Serial part: scoring runs one tile at a time in this main thread.
        for tile in batch:
            pixels = pixels_by_key[tile.key]
            cov = cov_by_key[tile.key]
            # >> Either download failed -> a FAILED row in tile_stats and no panel rows
            # >> for this tile.
            # >> Gotcha: that tile's chip_ids are then missing, so this period's panel is
            # >> not balanced.
            # >> Also, coverage.py has no retry of its own (only the imagery path retries
            # >> in gee.py).
            if pixels is None or cov is None:
                # Failed even after gee.py's internal retry — record and move on;
                # the analysis treats absent tiles as missing, never as zeros.
                tile_stats.append({"period": spec.label, "tile_key": tile.key,
                                   "status": "FAILED",
                                   "fetch_s": fetch_s / max(len(batch), 1)})
                log.error("[%s] tile %s failed (pixels=%s cov=%s)", spec.label,
                          tile.key, pixels is not None, cov is not None)
                continue

            # Axis-order guard (OPTIMIZATIONS.md P2). Two conventions meet here:
            #   (H, W, B) numpy/TF — what every line below assumes
            #   (B, H, W) rasterio  — what the cache GeoTIFFs store
            # gee.load_tile and coverage._load transpose back, but nothing asserts it. A
            # transposed array does NOT raise: ensure_tile_shape would read height=13,
            # reflect-pad to 576x576 and return plausible garbage. Cheap check, silent
            # scientific error avoided.
            # >> 13 bands for S2L1C: B1..B12 including B8A and B10.
            n_bands = len(data_config.bands)
            assert pixels.ndim == 3 and pixels.shape[-1] == n_bands, (
                f"imagery for {tile.key} is {pixels.shape}, expected (H, W, {n_bands}) — "
                "axis order wrong or cache file corrupt")
            assert cov.ndim == 3 and cov.shape[-1] == 2, (
                f"coverage for {tile.key} is {cov.shape}, expected (H, W, 2)")
            assert pixels.shape[:2] == cov.shape[:2], (
                f"imagery {pixels.shape[:2]} and coverage {cov.shape[:2]} disagree on grid "
                f"for {tile.key} — chip_coverage_stats would read the wrong window")

            t1 = time.perf_counter()
            # >> Normalise: raw values are reflectance x 10,000 -> divide, clip to [0, 1],
            # >> float32.
            # normalization identical to InferenceEngine.predict_on_tile_pixels
            normed = np.clip(pixels / 10000.0, 0, 1).astype(np.float32)
            # >> cut_chips: (576, 576, 13) -> chips (529, 48, 48, 13) + GeoDataFrame of
            # >> 529 chip squares built from the tile's lat/lon bounds (EPSG:4326). Order:
            # >> column-major.
            chips, chip_geoms = tile_utils.cut_chips(
                normed, tile.geometry.bounds, chip_size, stride, crs="epsg:4326")
            # >> Model call: (529, 48, 48, 13) -> raw (529, 6), one sigmoid score per CNN
            # >> member.
            # Keep the RAW (n, n_members) output before it is collapsed to a mean.
            # `prob` below stays bit-identical to what bulk_predict would produce.
            raw_preds = engine._model_infer(chips)
            # >> prob = mean over the 6 members (axis 1) -> (529,), bit-identical to
            # >> upstream.
            probs = engine._probs_from_predictions(raw_preds)
            # >> Keep the (529, 6) member scores for columns p_m0..p_m5.
            members = member_probs(raw_preds)
            infer_s = time.perf_counter() - t1

            # >> Re-create the chip corners; width = shape[1] (columns), height = shape[0]
            # >> (rows).
            windows = list(chip_windows(pixels.shape[1], pixels.shape[0],
                                        chip_size, stride))
            # Hard guarantee that our window order matches cut_chips' order:
            assert len(windows) == len(chip_geoms) == len(probs), (
                f"window/chip count mismatch on {tile.key}: "
                f"{len(windows)} vs {len(chip_geoms)} vs {len(probs)}")

            # >> Per-tile fill-zero check (see fill_zero_mismatch above).
            mismatch = fill_zero_mismatch(pixels, cov)
            # >> One panel row per chip; chip i <-> window i <-> prob i (same column-major
            # >> order).
            for i, ((row, col), prob) in enumerate(zip(windows, probs)):
                # >> Per-chip cloud summary from the 48 x 48 window of the coverage
                # >> counts.
                clear_frac, ncl_mean, ncl_min, ntot_mean = chip_coverage_stats(
                    cov, row, col, chip_size)
                # >> chip_id = "<tile_key>:c<i>", e.g. "552:12:10.0:19:-15:-263:c17". Same
                # >> id in every period because tiling depends only on the AOI -> balanced
                # >> panel (same chip set every period).
                rec = {
                    "chip_id": f"{tile.key}:c{i}",
                    "tile_key": tile.key,
                    # >> row/col = top-left pixel of the chip inside the 576 px tile (row
                    # >> 0 = north edge).
                    "row": row, "col": col,
                    # >> prob = ensemble-mean score in [0, 1]; NOT thresholded.
                    "prob": float(prob),
                    # >> clear_frac: share of chip pixels with >= 1 clear scene;
                    # >> n_clear_*: clear scenes per pixel (mean, min); n_total_mean:
                    # >> scenes covering the pixel at all, clear or not.
                    "clear_frac": clear_frac,       # the cloud flag (1 = fully observed)
                    "n_clear_mean": ncl_mean,       # composite precision proxy
                    "n_clear_min": ncl_min,
                    "n_total_mean": ntot_mean,
                    # >> geometry = chip square in EPSG:4326 (lat/lon).
                    "geometry": chip_geoms.geometry.iloc[i],
                }
                # Raw per-member probabilities: p_m0 … p_m{k-1}, ordered as the model's
                # Concatenate output. Names recorded once in the ensemble_members column.
                if members is not None:
                    for m in range(members.shape[1]):
                        rec[f"p_m{m}"] = float(members[i, m])
                rows.append(rec)

            # >> Per-tile diagnostics row -> tile_stats CSV. tile_clear_frac /
            # >> tile_n_clear_mean use the full 576 px tile including the pad.
            tile_stats.append({
                "period": spec.label, "tile_key": tile.key, "status": "OK",
                "n_chips": len(windows),
                "tile_clear_frac": float(np.mean(cov[..., 0] > 0)),
                "tile_n_clear_mean": float(np.mean(cov[..., 0])),
                "fill0_mismatch": mismatch,
                "fetch_s": fetch_s / max(len(batch), 1),  # amortized batch fetch
                "infer_s": infer_s,
            })
        log.info("[%s] tiles %d-%d/%d done", spec.label, b0 + 1,
                 min(b0 + max_workers, len(tiles)), len(tiles))

    # ------------------------------------------------------------------ write
    # >> Build the table once, at the end of the period; CRS EPSG:4326 (lat/lon).
    # >> If every tile failed, rows is empty and this raises -> run_trials marks the
    # >> period FAILED.
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # >> Period metadata columns (same value on every row).
    gdf["period"] = spec.label
    gdf["cadence"] = spec.cadence
    gdf["start"] = spec.start
    gdf["end"] = spec.end
    # >> Count member columns: 6 for the production ensemble, 0 if member_probs gave None.
    n_members = sum(1 for c in gdf.columns if c.startswith("p_m"))
    gdf["n_members"] = n_members
    # >> Add provenance columns (constant per file; Parquet dictionary-encodes them
    # >> cheaply).
    for k, v in prov.items():
        gdf[k] = v
    # >> FILE WRITE: GeoParquet via pyarrow, written once per period. Why Parquet:
    # >> columnar (read only the columns you need), typed (floats stay floats), compressed
    # >> (snappy by default); geometry stored as WKB (standard binary geometry) with the
    # >> CRS in the file metadata.
    gdf.to_parquet(panel_path, index=False)

    # >> Per-tile stats go to a sibling folder: <out_root>/tiles/.
    stats_dir = out_dir.parent / "tiles"
    stats_dir.mkdir(parents=True, exist_ok=True)
    # >> FILE WRITE: tile_stats_<label>.csv, one row per tile (status, timing, coverage,
    # >> fill0).
    pd.DataFrame(tile_stats).to_csv(
        stats_dir / f"tile_stats_{spec.label}.csv", index=False)

    # >> Positives for eyeballing in QGIS: prob > pred_threshold (0.4 by default).
    det_dir = out_dir.parent / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    pos = gdf[gdf["prob"] > pred_threshold]
    if len(pos):
        # >> FILE WRITE: detections_<label>.geojson (lat/lon). Only when there is >= 1
        # >> positive; gotcha: otherwise an old file from an earlier run is left in place.
        pos.to_file(det_dir / f"detections_{spec.label}.geojson", driver="GeoJSON")

    log.info("[%s] wrote %d chips (%d positive @ %.2f, %d ensemble members, "
             "schema v%d) -> %s",
             spec.label, len(gdf), len(pos), pred_threshold, n_members,
             PANEL_SCHEMA_VERSION, panel_path)
    return panel_path


# --------------------------------------------------------------------------- #
# fetch wrappers (thread-pool safe: never raise, return None on failure)      #
# --------------------------------------------------------------------------- #
# >> What: downloads (or loads from cache) one tile's imagery; never crashes the pool.
# >> Calls upstream get_tile_data: cached GeoTIFF, else EE computePixels (google.api_core
# >> retry, 240 s).
# >> Out: (576, 576, 13) array, raw reflectance x 10,000; None on any error (logged).
# >> Runs inside the thread pool of build_period_panel.
def _fetch_pixels_safe(engine: InferenceEngine, tile) -> Optional[np.ndarray]:
    try:
        return engine.data_extractor.get_tile_data(tile)  # cache-aware
    except Exception as e:
        log.warning("pixel fetch failed for %s: %s", tile.key, e)
        return None


# >> What: same idea for cloud counts, via CoverageExtractor.get_tile_counts
# >> (coverage.py).
# >> Out: (576, 576, 2) float32 [N_CLEAR, N_TOTAL]; None on any error (logged).
def _fetch_cov_safe(cov_extractor: CoverageExtractor, tile) -> Optional[np.ndarray]:
    try:
        return cov_extractor.get_tile_counts(tile)  # cache-aware
    except Exception as e:
        log.warning("coverage fetch failed for %s: %s", tile.key, e)
        return None
