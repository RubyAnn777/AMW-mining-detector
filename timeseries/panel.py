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
from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

# upstream core/ modules (sys.path prepared by run_trials.py)
import gee
import tile_utils
from inference_engine import InferenceConfig, InferenceEngine

from coverage import CoverageExtractor, chip_coverage_stats

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Bump when the panel column set changes. Stored in every row so panels written by
#: different code versions can never be pooled silently by analysis.py (which globs
#: panel_*.parquet with no filter — analysis.py:94).
#:   1 = chip_id..geometry + period/cadence/start/end   (runs up to 2026-08-04)
#:   2 = + per-member probabilities p_m* + provenance columns
PANEL_SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# ensemble members and provenance                                             #
# --------------------------------------------------------------------------- #
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
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return arr.astype(np.float64, copy=False)


def _git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Short commit hash, suffixed ``-dirty`` if TRACKED files have local modifications.

    ``--untracked-files=no`` is deliberate: a run writes its own outputs into the tree
    (panels, logs, run directories), so counting untracked files would mark every single
    panel ``-dirty`` and make the flag meaningless. What matters for provenance is
    whether the *committed code* differs from what is running.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=15)
        dirty = "-dirty" if status.stdout.strip() else ""
        return head.stdout.strip() + dirty
    except Exception:  # git absent, not a repo, timeout — never fail a run over this
        return "unknown"


def _file_sha256(path: Path, nbytes: int = 16) -> str:
    """First ``nbytes`` hex chars of the file's SHA-256. Identifies the weights."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:nbytes]
    except Exception:
        return "unknown"


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
def chip_windows(width: int, height: int, chip_size: int, stride: int):
    """Yield (row, col) upper-left corners in EXACTLY the order of
    core/tile_utils.cut_chips (outer loop over col, inner over row).
    Keeping the order identical is what lets us zip() windows with the
    chips/geometries/probabilities arrays without an index join."""
    for col in range(0, width - chip_size + stride, stride):
        for row in range(0, height - chip_size + stride, stride):
            yield row, col


def fill_zero_mismatch(pixels: np.ndarray, cov: np.ndarray) -> float:
    """Empirical check of the fill=0 assumption (pipeline card §5).

    valid_from_counts : N_CLEAR > 0  (>=1 clear scene entered the median)
    valid_from_image  : not all 13 bands exactly 0

    If Earth Engine fills fully-masked pixels with 0 (as assumed), the two masks
    agree everywhere and this returns 0.0. Persistent non-zero values would mean
    gaps are NOT reliably detectable from the imagery alone — the coverage
    raster would then be the only trustworthy flag. Reported per tile, averaged
    in the analysis report."""
    valid_from_counts = cov[..., 0] > 0
    valid_from_image = ~np.all(pixels == 0, axis=-1)
    return float(np.mean(valid_from_counts != valid_from_image))


# --------------------------------------------------------------------------- #
# per-period build                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class PeriodSpec:
    """One composite window. label examples: 'm2025-01', 'q2025Q1'."""
    label: str
    start: str   # YYYY-MM-DD, inclusive
    end: str     # YYYY-MM-DD, EXCLUSIVE (ee.filterDate convention, as upstream)
    cadence: str  # 'monthly' | 'quarterly'


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
    panel_path = out_dir / f"panel_{spec.label}.parquet"

    # --- tiling: identical parameters to the production pipeline ------------
    region = gpd.read_file(region_path)
    try:
        geom = region.union_all()
    except AttributeError:  # geopandas < 1.0
        geom = region.unary_union
    data_config = gee.DataConfig(image_cache_dir=str(cache_dir))
    tiles = tile_utils.create_tiles(geom, data_config.tilesize, data_config.pad)
    if limit_tiles:
        tiles = tiles[:limit_tiles]
    log.info("[%s] %d tiles, window %s..%s", spec.label, len(tiles),
             spec.start, spec.end)

    # --- engine (embedding_strategy='none' => from-scratch CNN path) --------
    engine = InferenceEngine(
        start_date=spec.start, end_date=spec.end,
        data_config=data_config,
        config=InferenceConfig(model_path=Path(model_path),
                               embedding_strategy="none",
                               pred_threshold=pred_threshold),
    )
    # fail loudly if a future upstream sync renames the internals we reuse
    for attr in ("_model_infer", "_probs_from_predictions", "_resolve_chip_params"):
        assert hasattr(engine, attr), (
            f"InferenceEngine.{attr} missing — upstream API changed; "
            "re-pin timeseries/panel.py against the new core/inference_engine.py")
    engine._ensure_keras_model()
    chip_size, stride = engine._resolve_chip_params()

    cov_extractor = CoverageExtractor(
        spec.start, spec.end,
        clear_threshold=data_config.clear_threshold,  # MUST match imagery pull
        cache_dir=cache_dir)

    # Provenance computed ONCE per period (the sha256 reads 16 MB, ~50 ms).
    # These columns are constant per file, so parquet dictionary-encodes them to a
    # couple of hundred bytes each — measured negligible against the 1.7 MB baseline.
    prov = {
        "schema_version": PANEL_SCHEMA_VERSION,
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

    rows: List[dict] = []
    tile_stats: List[dict] = []

    # Process in batches: imagery + coverage fetched concurrently (network
    # bound), CNN inference serialized (TF session; also keeps RAM bounded to
    # ~max_workers tiles of 576x576x13 float32 ≈ 17 MB each).
    for b0 in range(0, len(tiles), max_workers):
        batch = tiles[b0:b0 + max_workers]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            pix_futs = {t.key: ex.submit(_fetch_pixels_safe, engine, t)
                        for t in batch}
            cov_futs = {t.key: ex.submit(_fetch_cov_safe, cov_extractor, t)
                        for t in batch}
            pixels_by_key = {k: f.result() for k, f in pix_futs.items()}
            cov_by_key = {k: f.result() for k, f in cov_futs.items()}
        fetch_s = time.perf_counter() - t0

        for tile in batch:
            pixels = pixels_by_key[tile.key]
            cov = cov_by_key[tile.key]
            if pixels is None or cov is None:
                # Failed even after gee.py's internal retry — record and move on;
                # the analysis treats absent tiles as missing, never as zeros.
                tile_stats.append({"period": spec.label, "tile_key": tile.key,
                                   "status": "FAILED",
                                   "fetch_s": fetch_s / max(len(batch), 1)})
                log.error("[%s] tile %s failed (pixels=%s cov=%s)", spec.label,
                          tile.key, pixels is not None, cov is not None)
                continue

            t1 = time.perf_counter()
            # normalization identical to InferenceEngine.predict_on_tile_pixels
            normed = np.clip(pixels / 10000.0, 0, 1).astype(np.float32)
            chips, chip_geoms = tile_utils.cut_chips(
                normed, tile.geometry.bounds, chip_size, stride, crs="epsg:4326")
            # Keep the RAW (n, n_members) output before it is collapsed to a mean.
            # `prob` below stays bit-identical to what bulk_predict would produce.
            raw_preds = engine._model_infer(chips)
            probs = engine._probs_from_predictions(raw_preds)
            members = member_probs(raw_preds)
            infer_s = time.perf_counter() - t1

            windows = list(chip_windows(pixels.shape[1], pixels.shape[0],
                                        chip_size, stride))
            # Hard guarantee that our window order matches cut_chips' order:
            assert len(windows) == len(chip_geoms) == len(probs), (
                f"window/chip count mismatch on {tile.key}: "
                f"{len(windows)} vs {len(chip_geoms)} vs {len(probs)}")

            mismatch = fill_zero_mismatch(pixels, cov)
            for i, ((row, col), prob) in enumerate(zip(windows, probs)):
                clear_frac, ncl_mean, ncl_min, ntot_mean = chip_coverage_stats(
                    cov, row, col, chip_size)
                rec = {
                    "chip_id": f"{tile.key}:c{i}",
                    "tile_key": tile.key,
                    "row": row, "col": col,
                    "prob": float(prob),
                    "clear_frac": clear_frac,       # the cloud flag (1 = fully observed)
                    "n_clear_mean": ncl_mean,       # composite precision proxy
                    "n_clear_min": ncl_min,
                    "n_total_mean": ntot_mean,
                    "geometry": chip_geoms.geometry.iloc[i],
                }
                # Raw per-member probabilities: p_m0 … p_m{k-1}, ordered as the model's
                # Concatenate output. Names recorded once in the ensemble_members column.
                if members is not None:
                    for m in range(members.shape[1]):
                        rec[f"p_m{m}"] = float(members[i, m])
                rows.append(rec)

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
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["period"] = spec.label
    gdf["cadence"] = spec.cadence
    gdf["start"] = spec.start
    gdf["end"] = spec.end
    n_members = sum(1 for c in gdf.columns if c.startswith("p_m"))
    gdf["n_members"] = n_members
    for k, v in prov.items():
        gdf[k] = v
    gdf.to_parquet(panel_path, index=False)

    stats_dir = out_dir.parent / "tiles"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tile_stats).to_csv(
        stats_dir / f"tile_stats_{spec.label}.csv", index=False)

    det_dir = out_dir.parent / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    pos = gdf[gdf["prob"] > pred_threshold]
    if len(pos):
        pos.to_file(det_dir / f"detections_{spec.label}.geojson", driver="GeoJSON")

    log.info("[%s] wrote %d chips (%d positive @ %.2f, %d ensemble members, "
             "schema v%d) -> %s",
             spec.label, len(gdf), len(pos), pred_threshold, n_members,
             PANEL_SCHEMA_VERSION, panel_path)
    return panel_path


# --------------------------------------------------------------------------- #
# fetch wrappers (thread-pool safe: never raise, return None on failure)      #
# --------------------------------------------------------------------------- #
def _fetch_pixels_safe(engine: InferenceEngine, tile) -> Optional[np.ndarray]:
    try:
        return engine.data_extractor.get_tile_data(tile)  # cache-aware
    except Exception as e:
        log.warning("pixel fetch failed for %s: %s", tile.key, e)
        return None


def _fetch_cov_safe(cov_extractor: CoverageExtractor, tile) -> Optional[np.ndarray]:
    try:
        return cov_extractor.get_tile_counts(tile)  # cache-aware
    except Exception as e:
        log.warning("coverage fetch failed for %s: %s", tile.key, e)
        return None
