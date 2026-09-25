#!/usr/bin/env python3
"""
Cumulative mine-scar masks with temporal persistence.

Produces one **onset raster** per UTM/lat band group: each pixel holds the year
its mining was confirmed, or 0. Every cumulative derives from it by
``onset != 0 & onset <= Y``, the raster analogue of the detection layer's
first-detection product, and it cannot decrease by construction.

Three things combine, and the order matters:

1. **Pixel-level persistence.** A pixel's mask onset is the first year it is
   masked *and* corroborated within the window -- the same ``k``-of-``n`` rule the
   detections use (recipe A: k=2, window=2), so the two halves admit a site on one
   definition. Only years whose window has closed can be onsets.

2. **Attribution to confirmed detections.** Detections are authoritative for
   *when*, masks for *how much*. Mask area is kept only where a confirmed
   detection could plausibly have generated it: bounded geodesic growth through
   the mask from confirmed detections, capped at the prior-implied extent.

   The cap is what makes rejection bite. Keeping whole connected components
   instead lets one surviving detection retain a 4,636 ha blob (7.6% of mask area
   in a Tapajós window); clipping to detection footprints instead discards the
   14.4% of per-period mask area that legitimately sits outside one. The cap is
   the pipeline's own bound: with ``penalty = -(dist/prior_sigma)^2``, a pixel
   cannot clear threshold beyond ``prior_sigma*sqrt(max_logit)``. Measured on 2023
   that is 42.7 px against a furthest observed mask pixel of 33.1 px, so it clips
   no real scar.

3. **Monotonicity.** Both inputs to a year -- persistent pixels and confirmed
   detections -- only grow with Y, so the admitted set only grows, and a pixel's
   onset is the first year it is admitted.

Years are stacked by windowed reads at integer lattice offsets. Per-year band
mosaics have different extents (each is the snapped union of that year's tiles)
but share the global grid, so no resampling is involved.

Usage:
    python sam2_persistence.py --years 2018 2019 2020 2021 2022 2023 2024 2025 \
        --group utm21_lat_-8_0
"""
from __future__ import annotations

import argparse
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import scipy.ndimage as ndi
from rasterio.windows import Window
from skimage.graph import MCP_Geometric

# >> Gotcha: the package-import branch below does not import encode_period, BASE or MODEL;
# >> encode_period is redefined further down, but main()'s defaults need BASE/MODEL, so
# >> only the plain-script branch (except ImportError) actually works for the CLI.
try:
    from .persistence import PersistenceConfig, Period
    from .sam2_logits import MASK_NODATA
except ImportError:
    from periods import Period, encode_period
    from pipeline_config import BASE, MODEL
    from persistence import PersistenceConfig
    from sam2_logits import MASK_NODATA

# >> FILE OVERVIEW: turns yearly SAM2 mask mosaics into ONE "onset raster" per band group
# >> (upstream AMW file). Onset per pixel = the first period in which that pixel's mining
# >> is confirmed, stored as a period code (year like 2021, or quarter code like 20251 =
# >> Q1 2025), 0 = never. Any "cumulative mined area up to year Y" is then onset != 0 &
# >> onset <= Y. Runs after sam2_build_cog.py (reads mining_mask_*_<group>_epsg4326.tif
# >> COGs) and after detection persistence (reads the cumulative detections GeoJSON with
# >> onset/status). Called by scripts/pipeline.py stage_persist_masks (once per group,
# >> then --mosaic).
REPO = Path(__file__).resolve().parent.parent
# >> Same 0.00009-degree lattice as sam2_build_cog.py; all inputs share it, so no
# >> resampling.
GRID_RES = 0.00009
# >> Must match MaskConfig.prior_sigma (12 px) and logit clamp (16) used when masks were
# >> made.
PRIOR_SIGMA = 12.0
LOGIT_CLAMP = 16.0

#: No confirmed mining. Unobserved ground reads the same: for a cumulative
#: product "not confirmed" is the answer either way, and a year's observed
#: footprint is recoverable from that year's own mask.
NO_ONSET = 0

# >> Ways to decide which mask pixels belong to a confirmed detection; only bounded_growth
# >> is used in production, the rest are for comparison.
ATTRIBUTION_RULES = ("bounded_growth", "nearest_seed", "components", "clip")


# >> Settings for the mask persistence rule. Plain dataclass (settings container).
# >> k-of-window: a pixel must be masked in >= k of `window` consecutive years (2 of 2).
# >> cap_px = how far (pixels) mask area may extend from a confirmed detection.
# >> window_px = processing block size (2048 px); unrelated to the year window.
@dataclass
class MaskPersistenceConfig:
    """The rule, and how mask area is attributed to confirmed detections."""

    #: Inherited so the mask and detection halves cannot drift apart.
    k: int = PersistenceConfig.k
    window: int = PersistenceConfig.window

    #: How mask area is tied to confirmed detections. ``bounded_growth`` is the
    #: chosen rule; the others exist to be measured against it.
    attribution: str = "bounded_growth"

    #: Growth cap in pixels. ``None`` derives ``prior_sigma*sqrt(max logit)``
    #: from the data, falling back to the clamp-implied ceiling.
    cap_px: Optional[float] = None

    #: Processing window. Attribution is spatial, so windows are read with a halo
    #: of ``cap_px`` and only their interior is written.
    window_px: int = 2048

    prior_sigma: float = PRIOR_SIGMA

    # >> Maximum reach of the spatial prior: 12 * sqrt(16) = 48 px (~480 m).
    # >> Beyond that distance the prior penalty alone exceeds the largest possible logit.
    def fallback_cap_px(self) -> float:
        """Hard ceiling from the logit clamp, independent of any data."""
        return self.prior_sigma * math.sqrt(LOGIT_CLAMP)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

#: Runs contributing mask to a band group. The Andes supplemental is a second
#: pass at a lower raw threshold over ground *inside* Amazon_ACA, so its masks
#: are additional coverage of the same band groups, not separate regions --
#: omitting it left utm18 lat[-16,-8] 86% below the published product.
# >> Folder-name globs for runs feeding a group: the main Amazon_ACA annual runs at t0.43,
# >> plus the Andes supplemental. {y} is the year. Note annual runs use 12-31 as end date.
RUN_PATTERNS = ("Amazon_ACA*_{y}-01-01_{y}-12-31_t0.43*",
                "andes_supplemental*_{y}-01-01_{y}-12-31")


# >> Finds each year's mask COGs for one group, across all matching run folders. Output:
# >> {year: [paths to mining_mask_*_<group>_epsg4326.tif]}; years with none omitted.
# >> Gotcha: this function is defined twice, identically; the second copy wins.
def band_group_paths(sam2_root: Path, years: Sequence[int], group: str,
                     patterns: Sequence[str] = RUN_PATTERNS
                     ) -> Dict[int, List[Path]]:
    """Per-year mask mosaics for one UTM/lat band group, across runs.

    A year maps to *every* run's mosaic for that group; the driver unions them.
    Detections are deduplicated where the runs overlap, but masks need no
    equivalent step -- the union is idempotent, so a pixel masked by both runs is
    simply masked.
    """
    out: Dict[int, List[Path]] = {}
    for year in years:
        found: List[Path] = []
        for pattern in patterns:
            for hit in glob.glob(str(sam2_root / pattern.format(y=year)
                                     / "cog_outputs"
                                     / f"mining_mask_*_{group}_epsg4326.tif")):
                found.append(Path(hit))
        if found:
            out[year] = sorted(found)
    return out


# >> Duplicate of the function above (same code). Python keeps this later definition.
def band_group_paths(sam2_root: Path, years: Sequence[int], group: str,
                     patterns: Sequence[str] = RUN_PATTERNS
                     ) -> Dict[int, List[Path]]:
    """Per-year mask mosaics for one UTM/lat band group, across runs.

    A year maps to *every* run's mosaic for that group; the driver unions them.
    Detections are deduplicated where the runs overlap, but masks need no
    equivalent step -- the union is idempotent, so a pixel masked by both runs is
    simply masked.
    """
    out: Dict[int, List[Path]] = {}
    for year in years:
        found: List[Path] = []
        for pattern in patterns:
            for hit in glob.glob(str(sam2_root / pattern.format(y=year)
                                     / "cog_outputs"
                                     / f"mining_mask_*_{group}_epsg4326.tif")):
                found.append(Path(hit))
        if found:
            out[year] = sorted(found)
    return out


# >> Turns a period tag into a sortable integer: "2024" -> 2024, "Q125" -> 20251. Written
# >> as the pixel value of the onset raster (uint16, max 65535, fits). Gotcha: annual 2026
# >> (2026) sorts BEFORE Q1 2025 (20251), so codes are only chronological if every quarter
# >> present is later than every year present.
def encode_period(tag: str) -> int:
    """Sortable uint16 code for a period tag: ``2024`` -> 2024, ``Q125`` -> 20251.

    Chronological under plain comparison (2024 < 20251 < 20252 < 20261), so any
    cumulative is ``0 < onset <= code``. Years and quarters share one raster,
    which is what makes the year-boundary supersede structural: quarters are only
    emitted for years the rule cannot yet resolve, so recomputing after the next
    annual lands replaces them with the confirmed year automatically.
    """
    period = Period.parse(tag)
    return period.year if period.is_annual else period.year * 10 + period.quarter


# >> Finds each quarter's diff-mask COGs (run folders "*growth_<tag>") for one group.
# >> Output: {tag: [paths]}. Quarterly runs mask only NEW locations (patch_diffs).
def quarter_group_paths(sam2_root: Path, quarters: Sequence[str], group: str
                        ) -> Dict[str, List[Path]]:
    """Per-quarter diff-mask mosaics for one band group.

    Quarterly SAM2 runs segment the *increment* (patch_diffs), not the period, so
    these cover only locations new to that quarter.
    """
    out: Dict[str, List[Path]] = {}
    for tag in quarters:
        span = Period.parse(tag).date_span
        hits = [Path(h) for h in glob.glob(str(
            sam2_root / f"*growth_{tag}" / "cog_outputs"
            / f"mining_mask_{span}_{group}_epsg4326.tif"))]
        if hits:
            out[tag] = sorted(hits)
    return out


# >> Computes one pixel grid that covers all inputs (union of their extents). Output:
# >> (affine transform, width px, height px) in EPSG:4326. Works in integer lattice
# >> indices: c/f = x/y of the top-left corner from each file's affine transform.
def union_grid(paths: Sequence[Path]) -> Tuple[rasterio.Affine, int, int]:
    """Common grid covering every input, exploiting the shared lattice.

    Origins are integer multiples of GRID_RES, so the union is an integer pixel
    box and every input aligns to it without resampling.
    """
    i0 = j0 = math.inf
    i1 = j1 = -math.inf
    for path in paths:
        with rasterio.open(path) as ds:
            # >> Integer column index of the file's west edge / row index of its north
            # >> edge.
            ci = round(ds.transform.c / GRID_RES)
            fj = round(ds.transform.f / GRID_RES)
            i0, j0 = min(i0, ci), min(j0, fj - ds.height)
            i1, j1 = max(i1, ci + ds.width), max(j1, fj)
    width, height = int(i1 - i0), int(j1 - j0)
    # >> Affine: pixel size 0.00009 deg, north-up (negative y step), top-left at (i0, j1).
    transform = rasterio.Affine(GRID_RES, 0, i0 * GRID_RES,
                                0, -GRID_RES, j1 * GRID_RES)
    return transform, width, height


# >> Measures the real growth cap from data: prior_sigma * sqrt(largest logit seen),
# >> sampling up to 40 per-tile -logits.tif files per run folder (two levels above each
# >> COG). Falls back to 48 px (clamp-based) if no logits are found.
def derive_cap_px(paths: Sequence[Path], config: MaskPersistenceConfig,
                  sample: int = 40) -> float:
    """``prior_sigma*sqrt(max logit)``, from the logits beside the masks."""
    best = -math.inf
    for path in paths:
        tile_dir = path.parent.parent
        logits = sorted(glob.glob(str(tile_dir / "*-logits.tif")))
        if not logits:
            continue
        # >> Evenly spaced sample of the logit files (every step-th, at most `sample`).
        step = max(1, len(logits) // sample)
        for lp in logits[::step][:sample]:
            with rasterio.open(lp) as ds:
                arr = ds.read(1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                best = max(best, float(finite.max()))
    if not math.isfinite(best) or best <= 0:
        return config.fallback_cap_px()
    return config.prior_sigma * math.sqrt(best)


# >> Reads a window of the union grid out of one COG, shifting by whole pixels.
# >> Outside the file's own extent it returns MASK_NODATA (2) via boundless read.
# >> Output: 2-D uint8 (window.height, window.width).
def read_on_grid(path: Path, transform: rasterio.Affine, window: Window
                 ) -> np.ndarray:
    """Read ``window`` of the union grid from ``path``, padding where absent."""
    with rasterio.open(path) as ds:
        # >> Pixel offset between this file and the union grid (integer, thanks to the
        # >> lattice).
        di = round((transform.c - ds.transform.c) / GRID_RES)
        dj = round((ds.transform.f - transform.f) / GRID_RES)
        src = Window(window.col_off + di, window.row_off + dj,
                     window.width, window.height)
        return ds.read(1, window=src, boundless=True, fill_value=MASK_NODATA)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

# >> Core persistence rule per pixel. Input: bool stack (n_years, h, w), True = masked.
# >> Output int16 (h, w): index of the first year i where the pixel is masked in year i
# >> AND masked in >= k of years i..i+window-1; -1 = never. Last window-1 years cannot be
# >> onsets. Gotcha: indices are positions in the list of available years, so a missing
# >> year makes the "window" span two calendar years silently.
def mask_onset_index(stack: np.ndarray, config: MaskPersistenceConfig
                     ) -> np.ndarray:
    """First year index at which each pixel's mask is corroborated.

    ``stack`` is (n_years, h, w) boolean. Returns int16 indices, -1 where never.
    Only years whose window has closed are candidates, matching
    ``persistence.resolvable_periods``.
    """
    n = len(stack)
    # >> Start with -1 everywhere ("no onset yet").
    onset = np.full(stack.shape[1:], -1, dtype=np.int16)
    counts = stack.astype(np.uint8)
    # >> Slide the window over years in chronological order; first hit wins (onset < 0
    # >> check).
    for i in range(n - config.window + 1):
        # >> How many of the years in this window the pixel was masked; corroborated if >=
        # >> k.
        corroborated = counts[i:i + config.window].sum(axis=0) >= config.k
        take = stack[i] & corroborated & (onset < 0)
        onset[take] = i
    return onset


# >> Keeps only mask pixels that a confirmed detection could have produced.
# >> Input: mask (h, w) bool, seeds (h, w) bool = rasterized detection polygons, cap_px.
# >> Output: bool (h, w). Rule chosen by config.attribution (see ATTRIBUTION_RULES).
# >> Why: detections decide WHEN a site counts; masks decide HOW MUCH area.
def attribute(mask: np.ndarray, seeds: np.ndarray, config: MaskPersistenceConfig,
              cap_px: float) -> np.ndarray:
    """Mask area a confirmed detection could plausibly have generated."""
    if not mask.any() or not seeds.any():
        return np.zeros_like(mask)

    rule = config.attribution
    # >> "clip": only mask pixels inside detection footprints (too strict, loses ~14%
    # >> area).
    if rule == "clip":
        return mask & seeds
    # >> "components": whole 8-connected blobs touching a seed (too loose: can keep huge
    # >> blobs).
    if rule == "components":
        labels, _ = ndi.label(mask, np.ones((3, 3)))
        keep = np.unique(labels[seeds & mask])
        return np.isin(labels, keep[keep > 0])
    # >> "bounded_growth" (production): grow outward from seed pixels, moving only through
    # >> mask pixels, and keep pixels whose path length from a seed is <= cap_px.
    if rule == "bounded_growth":
        # Geodesic distance *through the mask* from any seed. MCP_Geometric gives
        # true euclidean path length, so the cap is comparable to the analytic
        # prior bound; iterative dilation would give chebyshev and over-reach
        # diagonally by up to sqrt(2).
        # >> Cost 1 per step inside the mask, infinite outside (growth cannot cross
        # >> non-mask).
        costs = np.where(mask, 1.0, np.inf)
        # >> Start points = mask pixels under a detection; (n, 2) array of (row, col).
        starts = np.argwhere(seeds & mask)
        if not len(starts):
            return np.zeros_like(mask)
        # >> skimage MCP_Geometric = minimum-cost-path solver; with fully_connected=True
        # >> it moves in 8 directions and charges sqrt(2) for diagonals, so dist is a true
        # >> Euclidean path length.
        mcp = MCP_Geometric(costs, fully_connected=True)
        # >> dist = cumulative cost from nearest start to every pixel (inf if
        # >> unreachable).
        dist, _ = mcp.find_costs(starts.tolist())
        # >> Keep reachable mask pixels within cap_px pixels of path distance.
        return mask & np.isfinite(dist) & (dist <= cap_px)
    # >> "nearest_seed": straight-line distance cap within seeded components (tighter
    # >> fallback).
    if rule == "nearest_seed":
        # Every mask pixel belongs to its nearest seed; a rejected detection
        # loses its own neighbourhood and nothing else. Seeds here are already
        # the retained set, so this reduces to a plain distance cap within the
        # component -- the strictly tighter fallback.
        dist = ndi.distance_transform_edt(~(seeds & mask))
        labels, _ = ndi.label(mask, np.ones((3, 3)))
        keep = np.unique(labels[seeds & mask])
        return mask & np.isin(labels, keep[keep > 0]) & (dist <= cap_px)
    raise ValueError(f"unknown attribution rule {rule!r}")


# >> Draws the chosen detection polygons into a boolean raster for one window.
# >> Selects rows with status in `statuses` and onset code <= through_code.
# >> Uses the GeoDataFrame spatial index to pre-filter, then rasterio.features.rasterize.
# >> Output: bool (window.height, window.width). Called by build_onset_raster.
def selected_detection_mask(dets: gpd.GeoDataFrame, sindex, through_code: int,
                            transform: rasterio.Affine, window: Window,
                            statuses: Tuple[str, ...] = ("confirmed",)
                            ) -> np.ndarray:
    """Rasterize detections selected at or before ``through_code``.

    ``statuses`` is ``("confirmed",)`` for annual onsets and includes
    ``"provisional"`` at the quarterly edge, where nothing is confirmed yet.
    """
    # >> Affine transform and lon/lat bounds of just this window.
    win_transform = rasterio.windows.transform(window, transform)
    bounds = rasterio.windows.bounds(window, transform)
    # >> R-tree lookup: detections whose bbox touches the window.
    hits = list(sindex.intersection(bounds))
    shape = (int(window.height), int(window.width))
    if not hits:
        return np.zeros(shape, dtype=bool)
    sub = dets.iloc[hits]
    sub = sub[sub["status"].isin(statuses) & (sub["_code"] <= through_code)]
    if sub.empty:
        return np.zeros(shape, dtype=bool)
    # >> Burn polygons as 1 on a 0 background (pixel centre inside polygon = burned).
    return rasterio.features.rasterize(
        ((g, 1) for g in sub.geometry if g is not None and not g.is_empty),
        out_shape=shape, transform=win_transform, fill=0, dtype="uint8"
    ).astype(bool)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# >> Writes the onset raster for one group: mask_onset_<group>_<rule>.tif, uint16,
# >> EPSG:4326, 512 px tiles, ZSTD, nodata 0. Processes the grid in 2048 px windows with a
# >> halo (extra border) of cap_px + 2 so growth near a window edge sees its neighbours.
# >> For each resolvable year in order: persistent pixels -> attribute to confirmed
# >> detections -> new pixels get that year. Then quarterly diffs (provisional allowed).
# >> Returns stats dict (windows, nonempty, onset_px, cap_px).
def build_onset_raster(mask_paths: Dict[int, List[Path]], detections: Path,
                       out_path: Path, config: MaskPersistenceConfig,
                       cap_px: Optional[float] = None,
                       quarter_paths: Optional[Dict[str, List[Path]]] = None
                       ) -> dict:
    """Write the onset-year raster for one band group.

    Windows are read with a ``cap_px`` halo and only their interior is written:
    attribution grows through the mask, so a pixel near a window edge must see
    the seeds and mask just outside it or growth would be truncated at an
    arbitrary boundary.
    """
    years = sorted(mask_paths)
    quarter_paths = quarter_paths or {}
    # >> Quarters sorted by their period code (chronological).
    quarters = sorted(quarter_paths, key=encode_period)
    flat = ([p for y in years for p in mask_paths[y]]
            + [p for q in quarters for p in quarter_paths[q]])
    # >> One common grid for every year and quarter input.
    transform, width, height = union_grid(flat)
    if cap_px is None:
        cap_px = config.cap_px or derive_cap_px(flat, config)
    # >> Halo in whole pixels; +2 is a safety margin for rounding/diagonals.
    halo = int(math.ceil(cap_px)) + 2

    # >> Years that can be onsets: all but the last window-1 (e.g. 2018..2024 of
    # >> 2018..2025).
    resolvable = years[:len(years) - config.window + 1]
    print(f"  grid {width:,} x {height:,}  cap {cap_px:.1f} px  halo {halo} px")
    print(f"  onset candidates: {resolvable[0]}-{resolvable[-1]} "
          f"(window {config.window} needs {config.window - 1} following year(s))")

    # >> Detections with their onset period, onset year and status
    # >> (confirmed/provisional/...).
    dets = gpd.read_file(detections, columns=["onset", "onset_year", "status"])
    # >> Convert each onset tag to the same integer code as the raster values.
    dets["_code"] = [encode_period(v) for v in dets["onset"]]
    # >> Build the spatial index once (used for every window).
    sindex = dets.sindex

    # >> Output GeoTIFF settings: uint16 period codes, 512 px internal tiles, ZSTD
    # >> compression, BIGTIFF only if needed. Plain tiled GeoTIFF here; --mosaic later
    # >> converts to COG.
    profile = dict(driver="GTiff", height=height, width=width, count=1,
                   dtype="uint16", crs="EPSG:4326", transform=transform,
                   nodata=NO_ONSET, tiled=True, blockxsize=512, blockysize=512,
                   compress="zstd", BIGTIFF="IF_SAFER")

    step = config.window_px
    n_win = math.ceil(width / step) * math.ceil(height / step)
    stats = dict(windows=n_win, nonempty=0, onset_px=0, cap_px=cap_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # >> File write: opened once, blocks written window by window.
    with rasterio.open(out_path, "w", **profile) as dst:
        done = 0
        # >> Loop order: rows (north->south) outer, columns (west->east) inner, step 2048
        # >> px.
        for row in range(0, height, step):
            for col in range(0, width, step):
                done += 1
                w = min(step, width - col)
                h = min(step, height - row)
                # padded read window, clipped to the grid
                pc, pr = max(0, col - halo), max(0, row - halo)
                pw = min(width, col + w + halo) - pc
                ph = min(height, row + h + halo) - pr
                padded = Window(pc, pr, pw, ph)

                # >> Stack shape (n_years, ph, pw): per year, OR over all runs' mosaics
                # >> (value 1 = mine; nodata 2 counts as not masked).
                stack = np.stack([
                    np.logical_or.reduce([read_on_grid(p, transform, padded) == 1
                                          for p in mask_paths[y]])
                    for y in years])

                # An empty annual stack does not mean an empty window: a
                # location first segmented in a quarter has no annual mask
                # anywhere near it, so the quarterly edge below must still run.
                onset = np.zeros((ph, pw), dtype=np.uint16)
                if stack.any():
                    # >> Per-pixel first corroborated year index (see mask_onset_index).
                    mask_onset = mask_onset_index(stack, config)
                    for idx, year in enumerate(resolvable):
                        # >> Pixels persistent by year idx (onset index <= idx): the set
                        # >> only grows with the year.
                        persistent = (mask_onset >= 0) & (mask_onset <= idx)
                        if not persistent.any():
                            continue
                        # >> Seeds = confirmed detections with onset <= this year.
                        seeds = selected_detection_mask(
                            dets, sindex, year, transform, padded)
                        admitted = attribute(persistent, seeds, config, cap_px)
                        # >> Only pixels without an earlier onset get this year (earliest
                        # >> confirmed year wins).
                        fresh = admitted & (onset == NO_ONSET)
                        onset[fresh] = year
                elif not quarters:
                    continue        # nothing annual, and no quarterly edge to add

                # Quarterly edge: no corroboration is possible yet, so a
                # quarter's diff mask is admitted wherever a provisional
                # detection of that quarter can reach it. Applied after the
                # annual years so a confirmed onset always wins -- which is also
                # why re-running after the next annual supersedes these.
                # >> Quarters: diff mask of that quarter, seeds may be provisional; never
                # >> overwrite a year.
                for tag in quarters:
                    code = encode_period(tag)
                    diff = np.logical_or.reduce(
                        [read_on_grid(p, transform, padded) == 1
                         for p in quarter_paths[tag]])
                    if not diff.any():
                        continue
                    seeds = selected_detection_mask(
                        dets, sindex, code, transform, padded,
                        statuses=("confirmed", "provisional"))
                    admitted = attribute(diff, seeds, config, cap_px)
                    fresh = admitted & (onset == NO_ONSET)
                    onset[fresh] = code

                # >> Cut the halo off: keep only this window's own interior before
                # >> writing.
                interior = onset[row - pr:row - pr + h, col - pc:col - pc + w]
                # >> Windows with no onset are not written; unwritten tiles read as nodata
                # >> 0.
                if interior.any():
                    stats["nonempty"] += 1
                    stats["onset_px"] += int((interior != NO_ONSET).sum())
                    dst.write(interior, 1, window=Window(col, row, w, h))
                if done % 200 == 0:
                    print(f"    {done:,}/{n_win:,} windows, "
                          f"{stats['onset_px']:,} px confirmed", flush=True)
    return stats


# >> Merges all per-group onset rasters into one COG with GDAL: gdalbuildvrt (plain
# >> mosaic, 0 = nodata) then gdal_translate -of COG (ZSTD, 512 px blocks, BIGTIFF, all
# >> CPUs). Groups do not overlap (a tile belongs to one group by its centre), so no merge
# >> rule.
def mosaic_onset_rasters(rasters: Sequence[Path], out_path: Path,
                         blocksize: int = 512) -> dict:
    """Merge the per-band onset rasters into one COG.

    Band groups partition the basin -- a tile belongs to exactly one, by its
    centre -- so the parts are disjoint and a plain mosaic is exact. No derived
    band, no pixel function: the expensive union happened upstream.
    """
    import subprocess
    import tempfile

    rasters = [Path(r) for r in rasters]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onset_mosaic_") as tmp:
        vrt = Path(tmp) / "onset.vrt"
        # >> Plain VRT mosaic; -srcnodata/-vrtnodata 0 so empty areas stay "no onset".
        subprocess.run(["gdalbuildvrt", "-srcnodata", str(NO_ONSET),
                        "-vrtnodata", str(NO_ONSET), str(vrt)]
                       + [str(r) for r in rasters], check=True,
                       capture_output=True, text=True)
        # >> Write the final COG (Cloud-Optimized GeoTIFF) to out_path.
        subprocess.run(["gdal_translate", str(vrt), str(out_path), "-of", "COG",
                        "-co", "COMPRESS=ZSTD", "-co", f"BLOCKSIZE={blocksize}",
                        "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
                        "-a_nodata", str(NO_ONSET)],
                       check=True, capture_output=True, text=True)
    with rasterio.open(out_path) as ds:
        return dict(width=ds.width, height=ds.height, parts=len(rasters))


# >> CLI. Two modes: (1) --group utm21_lat_-8_0: build that group's onset raster;
# >> (2) --mosaic OUT.tif: merge existing per-group rasters into one COG.
# >> Default detections: <BASE>/cumulative/Amazon_ACA_<MODEL>_detections.geojson.
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = MaskPersistenceConfig()
    ap.add_argument("--sam2_root", type=Path,
                    default=REPO / "data/outputs/sam2")
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(range(2018, 2026)))
    ap.add_argument("--quarters", nargs="*", default=[],
                    help="Quarterly diff runs to append at the provisional edge, "
                         "e.g. --quarters Q125 Q225")
    ap.add_argument("--mosaic", type=Path, default=None,
                    help="Merge existing per-band onset rasters into this COG "
                         "and exit; --group is then ignored")
    ap.add_argument("--group", default=None,
                    help="band group tag, e.g. utm21_lat_-8_0")
    ap.add_argument("--detections", type=Path,
                    default=BASE / "cumulative" / f"Amazon_ACA_{MODEL}_detections.geojson")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--attribution", choices=ATTRIBUTION_RULES,
                    default=defaults.attribution)
    ap.add_argument("--cap-px", dest="cap_px", type=float, default=None)
    ap.add_argument("--window-px", dest="window_px", type=int,
                    default=defaults.window_px)
    ap.add_argument("--k", type=int, default=defaults.k)
    ap.add_argument("--window", type=int, default=defaults.window)
    args = ap.parse_args()

    if args.mosaic:
        outdir = args.outdir or (REPO / "data/outputs/sam2/persistence_masks")
        # >> Gotcha: mosaic mode always globs *_bounded_growth.tif, whatever --attribution
        # >> says.
        parts = sorted(Path(outdir).glob("mask_onset_*_bounded_growth.tif"))
        if not parts:
            raise SystemExit(f"no per-band onset rasters in {outdir}")
        info = mosaic_onset_rasters(parts, args.mosaic)
        print(f"{info['parts']} band rasters -> {info['width']:,}x{info['height']:,}"
              f"  {args.mosaic}")
        return
    if not args.group:
        raise SystemExit("--group is required (or use --mosaic)")

    config = MaskPersistenceConfig(
        k=args.k, window=args.window, attribution=args.attribution,
        cap_px=args.cap_px, window_px=args.window_px)

    paths = band_group_paths(args.sam2_root, args.years, args.group)
    # >> Years with no mosaic found; the run continues without them (see index gotcha
    # >> above).
    missing = sorted(set(args.years) - set(paths))
    n_runs = sum(len(v) for v in paths.values())
    print(f"{args.group}: {len(paths)} of {len(args.years)} years, "
          f"{n_runs} run-mosaics"
          + (f"  (missing {missing})" if missing else ""))
    if len(paths) < config.window:
        raise SystemExit(f"need at least {config.window} years")

    outdir = args.outdir or (REPO / "data/outputs/sam2/persistence_masks")
    # >> Output: <outdir>/mask_onset_<group>_<attribution>.tif.
    out = outdir / f"mask_onset_{args.group}_{config.attribution}.tif"
    qpaths = quarter_group_paths(args.sam2_root, args.quarters, args.group)
    if args.quarters:
        missing = [q for q in args.quarters if q not in qpaths]
        print(f"  quarterly edge: {len(qpaths)} of {len(args.quarters)} runs"
              + (f"  (missing {missing})" if missing else ""))
    stats = build_onset_raster(paths, args.detections, out, config,
                               quarter_paths=qpaths)
    print(f"\n  {stats['onset_px']:,} confirmed px in "
          f"{stats['nonempty']:,}/{stats['windows']:,} windows -> {out}")


if __name__ == "__main__":
    main()
