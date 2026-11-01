#!/usr/bin/env python3
"""
Temporal persistence: which detections have earned a place in the cumulative record.

A detection enters the confirmed record only when corroborated by later
observation. Real mine scars are permanent and recur; cloud artefacts and
mosaic noise appear once and vanish. Requiring ``k`` detections within a window
of ``n`` consecutive periods removes most transient error while leaving genuine
features essentially untouched. Cumulatives need no threshold-tightening of
their own as a result.

Two properties make this safe to re-run at any time:

* **Anchored forward.** The window runs ``[Y, Y+n-1]``, never backwards -- a
  mine cannot have existed before it started. Onset is the first period that
  passes *its own* window, so a real 2022 onset is never backdated to a
  spurious 2019 cloud artefact.
* **Stateless.** Onset is a pure function of the full period stack. Nothing is
  carried between runs, so reprocessing cannot drift.

**Recipe A** is the chosen rule (``k=2``, ``window=2``, annual witnesses only);
see docs/design/persistence-planning.md, "Decision: recipe A". The rule stays
configurable so the alternatives can be re-measured, not because the choice is
open. The axes are the window length and whether quarterly periods count as
witnesses:

===========  ==========================  ==============================
             ``window=2`` (one year on)  ``window=3`` (two years on)
===========  ==========================  ==============================
annual only  A                           C  (the classic k=2 of n=3)
+ quarters   B                           D
===========  ==========================  ==============================

Periods too recent to resolve stay **provisional**, carrying the stricter
instantaneous threshold ``t_prov`` in place of temporal evidence they cannot
yet have. Provisional detections may be withdrawn; confirmed ones are not.
"""
from __future__ import annotations

# >> FILE OVERVIEW
# >> Turns per-period detections into the cumulative mining record: a chip is CONFIRMED
# >> only if it is detected again soon after (k=2 detections in a 2-year window).
# >> Pipeline: postprocessed_t0.43_*/ (+ Andes supplemental) -> THIS -> cumulative/
# >> (first-detection layer, per-period cumulatives, patch_diffs = SAM2 prompts) and
# >> cumulative_dissolved/ (polygons).
# >> Called by scripts/pipeline.py:stage_persist_detections.
# >> Upstream Amazon Mining Watch file (byte-identical to upstream).
import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
# >> pyproj.Geod = distances/areas measured on the curved Earth (WGS84 ellipsoid).
from pyproj import Geod

try:
    from .postprocess import (PostprocessConfig, centroid_key, dedupe_detections,
                              dissolve_patches)
except ImportError:
    from postprocess import (PostprocessConfig, centroid_key, dedupe_detections,
                             dissolve_patches)

# >> A location ID: (x, y) chip centre rounded to 5 decimals (postprocess.centroid_key).
LocKey = Tuple[float, float]

# Period vocabulary lives in periods.py -- a leaf module, so asking what dates a
# quarter spans does not import the geospatial stack. Re-exported here because
# `from persistence import Period` is the established import across the repo.
# >> Period = one annual (e.g. 2021) or quarterly (e.g. Q125 = Q1 2025) time slice.
# >> Gotcha: upstream periods.py date_span ends on 12-31 / 03-31 etc., and the EE end date
# >> is exclusive, so the last day of each period is never imaged.
from periods import (QUARTER_SPANS, QUARTER_TAG_RE,  # noqa: F401
                     Period, encode_period)
# >> Shared filename contract (base folder, model name, raw threshold tags).
from pipeline_config import ANDES_TAG, BASE, MODEL, RAW_TAG


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# >> The confirmation rule settings (Recipe A by default).
# >> k=2, window=2 years, annual periods only. Example: chip seen in 2021 and 2022 ->
# >> onset 2021, confirmed_in 2022. Seen only in 2021 -> not confirmed.
@dataclass
class PersistenceConfig:
    """The confirmation rule, and how the provisional edge is represented."""

    #: Detections required within the window, counting the onset period itself.
    #: k=3 would drop sites in cloud-wrecked years and short-lived operations
    #: that partly heal; k=2 is the working value.
    # >> k = 2 detections needed inside the window (the onset year counts as one).
    k: int = 2

    #: Window length in *years*, including the onset year. window=2 with k=2 is
    #: recipe A: seen in year Y and again in Y+1. Chosen over window=3 on
    #: timeliness and on review of the difference set -- see the planning doc,
    #: "Decision: recipe A". Less stringent than it looks: a cumulative record
    #: offers a mine every adjacent pair of years, not one designated window.
    # >> window = 2 years: onset year Y plus Y+1.
    window: int = 2

    #: Whether quarterly periods inside the window count as corroborating
    #: witnesses. Quarterly cloud loss is seasonal rather than random, so
    #: quarters bias towards dry-season-visible ground; with k=2 a single clear
    #: look suffices, which blunts but does not remove that bias.
    # >> False: quarterly mosaics cannot corroborate (cloudier, seasonal bias).
    use_quarterly_witnesses: bool = False

    #: Confirm using whatever periods exist, without waiting for the window to
    #: close. Safe: the witness set only grows towards a fixed endpoint, so
    #: confirmations accumulate monotonically and are never withdrawn -- but the
    #: answer keeps changing until the window closes.
    # >> False: wait until every year in the window exists before deciding.
    early_confirm: bool = False

    #: Also evaluate every shorter window and take the earliest onset. Shorter
    #: windows yield a strict subset of longer ones, so this only ever confirms
    #: sooner, never differently. Off by default.
    # >> False: do not also try shorter windows.
    nested: bool = False

    #: Instantaneous threshold standing in for temporal evidence on periods too
    #: recent to resolve. Count-matched against eventual confirmation, 0.50-0.53
    #: fits 2018-2023 better than the 0.55 inherited from the old cumulative.
    # >> t_prov 0.55 = stricter single-look threshold used for periods too recent to be
    # >> confirmed (e.g. the latest year, whose Y+1 does not exist yet).
    t_prov: float = 0.55

    #: Join key precision for "same location, different period".
    # >> Location-key precision: 5 decimals, same as postprocess.
    centroid_decimals: int = PostprocessConfig.centroid_decimals

    # >> Lists which periods count as witnesses for a given onset year.
    # >> With window=2 and onset 2021: annual periods with year 2021 or 2022.
    def witness_periods(self, onset: Period, available: Sequence[Period]) -> List[Period]:
        """Periods that may corroborate an onset at ``onset``."""
        last_year = onset.year + self.window - 1
        return [p for p in available
                if onset.year <= p.year <= last_year
                and (p.is_annual or self.use_quarterly_witnesses)]

    # >> True if all annual years the window needs are loaded (e.g. 2021 needs 2021+2022).
    # >> Only annual periods count here, even when quarterly witnesses are allowed.
    def window_is_closed(self, onset: Period, available: Sequence[Period]) -> bool:
        """True once every annual period the window needs is in hand."""
        needed = {onset.year + d for d in range(self.window)}
        have = {p.year for p in available if p.is_annual}
        return needed <= have


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

# >> Builds the path of one postprocessed detection file, e.g.
# >> postprocessed_t0.43_d5_3km_t-iso0.75/<stem>_2021-01-01_2021-12-31_t0.43_...geojson.
# >> Must match what postprocess.py:default_outpath + pipeline.py produce.
def detection_path(base: Path, period: Period, *, region_stem: str,
                   t_main: float, t_iso: float, k: int = 5,
                   isolation_km: float = 3.0) -> Path:
    """Path to a postprocessed single-period detection file."""
    tag = f"t{t_main:g}_d{k}_{isolation_km:g}km_t-iso{t_iso:g}"
    folder = base / f"postprocessed_{tag}"
    return folder / f"{region_stem}_{period.date_span}_{tag}.geojson"


# >> Path of the Andes supplemental detections for a period (written by
# >> scripts/geo_filter.py via pipeline stage_filter), or None if the file is absent.
def supplemental_path(base: Path, period: Period, *, stem: str,
                      threshold: str, subdir: str) -> Optional[Path]:
    """Path to the Andes supplemental detections for ``period``, if present."""
    path = base / subdir / f"{stem}_{threshold}_{period.date_span}.geojson"
    return path if path.is_file() else None


# >> Reads one period's detections and tags each chip with its location key.
# >> In: postprocessed GeoJSON (+ optional Andes file). Out: GeoDataFrame + _kx/_ky.
# >> Andes rows overlap the main run, so they are merged and deduped (keep highest
# >> confidence), exactly like the subregion seam duplicates.
def load_period(path: Path, decimals: int,
                supplemental: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Read one period's detections, keyed by rounded patch centroid.

    ``supplemental`` is unioned in and deduplicated. The Andes supplemental pass
    covers ground that lies *entirely inside* Amazon_ACA -- it is a second pass
    at a lower raw threshold (0.2 against 0.4) to catch the fainter Andean mines,
    not an extra region. So every supplemental detection may have a main-run
    twin: measured on 2024, all 229 main-run detections inside the boundary also
    appear in the supplemental set, differing by a median 8.8e-4 in confidence.
    That is the same disagreement the six-subregion seams produce, so the same
    rule applies -- keep the highest-confidence record per location.
    """
    gdf = gpd.read_file(path)
    if supplemental is not None:
        extra = gpd.read_file(supplemental)
        gdf = gpd.GeoDataFrame(
            # >> Stack main + Andes rows into one table.
            pd.concat([gdf, extra], ignore_index=True),
            crs=gdf.crs or extra.crs)
        # >> Drop repeated chip locations, keeping the highest-confidence copy.
        gdf = dedupe_detections(gdf, decimals=decimals, verbose=False)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # >> Attach the rounded-centre key used to match chips across periods.
    kx, ky = centroid_key(gdf, decimals=decimals)
    gdf = gdf.assign(_kx=kx, _ky=ky)
    return gdf


# >> The set of location keys present in one period (for fast 'was it seen?' checks).
def period_keys(gdf: gpd.GeoDataFrame) -> Set[LocKey]:
    return set(zip(gdf["_kx"], gdf["_ky"]))


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

# >> THE RULE: finds, for each location, its onset (first confirmed year).
# >> In: {period: set of location keys}. Out: {loc: (onset, confirmed_in, n_witnesses)}.
# >> Example (k=2, window=2): seen 2021 and 2022 -> (2021, 2022, 2).
# >> Seen 2019, 2021, 2022 -> 2019 fails its window (2020 missing); onset 2021.
# >> Only annual periods may be onsets. Stateless: same inputs always give same output.
def compute_onsets(
    detected: Dict[Period, Set[LocKey]],
    config: PersistenceConfig,
) -> Dict[LocKey, Tuple[Period, Period, int]]:
    """Confirmed onsets.

    Returns ``{location: (onset_period, confirmed_in, n_witnesses)}``, where
    ``confirmed_in`` is the earliest period at which the count reached ``k`` --
    the observation that settled it.

    Only annual periods are candidate onsets. A quarter is corroboration, not a
    starting point: quarterly mosaics are too cloud-affected to date an onset
    against, and dating from them would make onset depend on which quarters
    happened to be clear.
    """
    # >> Periods in time order (Period sorts by year then quarter).
    available = sorted(detected)
    # >> Normally just one window length (2); 'nested' would try 2..window.
    windows = range(2, config.window + 1) if config.nested else (config.window,)

    best: Dict[LocKey, Tuple[Period, Period, int]] = {}
    for span in windows:
        # >> A copy of the config with this window length.
        sub = PersistenceConfig(**{**config.__dict__, "window": span})
        # >> Data order: each ANNUAL period, oldest first, is a candidate onset.
        for onset in [p for p in available if p.is_annual]:
            # >> Skip onsets whose window is not complete yet (they stay provisional).
            if not config.early_confirm and not sub.window_is_closed(onset, available):
                continue
            witnesses = sub.witness_periods(onset, available)
            # >> Only chips detected IN the onset year can have that onset.
            for loc in detected[onset]:
                # >> Periods in the window where this location was also detected.
                seen = [p for p in witnesses if loc in detected[p]]
                if len(seen) < config.k:
                    continue
                # >> confirmed_in = period of the k-th sighting.
                settled = sorted(seen)[config.k - 1]
                prior = best.get(loc)
                # >> Keep the earliest onset if several years qualify.
                if prior is None or onset.sort_key < prior[0].sort_key:
                    best[loc] = (onset, settled, len(seen))
    return best


# >> Annual periods whose window can be judged now (e.g. with 2018-2025 loaded and
# >> window=2: 2018-2024; 2025 is not resolvable until 2026 exists).
def resolvable_periods(available: Sequence[Period],
                       config: PersistenceConfig) -> List[Period]:
    """Annual periods whose confirmation window can be evaluated."""
    return [p for p in available
            if p.is_annual
            and (config.early_confirm or config.window_is_closed(p, available))]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

# >> Builds the master layer: one row per location, with onset and status.
# >> Confirmed rows come from the t0.43 frames at their onset period; provisional rows
# >> come from t_prov (0.55/0.8) frames of unresolvable periods, if not already confirmed.
# >> Columns added: onset, onset_year, status, confirmed_in, n_witness.
# >> Output feeds main() -> <stem>_detections.geojson and every cumulative layer.
def build_first_detection_layer(
    frames: Dict[Period, gpd.GeoDataFrame],
    provisional: Dict[Period, gpd.GeoDataFrame],
    config: PersistenceConfig,
) -> gpd.GeoDataFrame:
    """One row per location: when mining was confirmed to have begun there.

    Every cumulative figure derives from this layer by filtering on ``onset``,
    which is why the per-period cumulatives are redundant with it. Confirmed
    rows never move; provisional rows may be withdrawn or promoted.
    """
    # >> Location-key sets per period, then run the rule.
    detected = {p: period_keys(g) for p, g in frames.items()}
    onsets = compute_onsets(detected, config)

    rows, seen = [], set()
    # >> Loop over periods in time order; 'seen' stops a location being added twice.
    for period in sorted(frames):
        if period not in {o[0] for o in onsets.values()}:
            continue
        gdf = frames[period]
        keys = list(zip(gdf["_kx"], gdf["_ky"]))
        # >> Rows of this period whose location has its onset exactly here.
        take = [i for i, key in enumerate(keys)
                if key not in seen and onsets.get(key, (None,))[0] == period]
        if not take:
            continue
        sub = gdf.iloc[take].copy()
        meta = [onsets[k] for k in (keys[i] for i in take)]
        # >> Label these rows as confirmed with their onset period.
        sub["onset"] = period.tag
        sub["onset_year"] = period.year
        sub["status"] = "confirmed"
        sub["confirmed_in"] = [m[1].tag for m in meta]
        sub["n_witness"] = [m[2] for m in meta]
        rows.append(sub)
        seen.update(keys[i] for i in take)

    # >> Second pass: provisional rows from recent, not-yet-resolvable periods.
    for period in sorted(provisional):
        gdf = provisional[period]
        keys = list(zip(gdf["_kx"], gdf["_ky"]))
        take = [i for i, key in enumerate(keys) if key not in seen]
        if not take:
            continue
        sub = gdf.iloc[take].copy()
        sub["onset"] = period.tag
        sub["onset_year"] = period.year
        sub["status"] = "provisional"
        sub["confirmed_in"] = None
        # >> Provisional = one sighting only (at the stricter t_prov threshold).
        sub["n_witness"] = 1
        rows.append(sub)
        seen.update(keys[i] for i in take)

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # >> Stack all pieces into one GeoDataFrame and drop the helper key columns.
    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True),
                           crs=rows[0].crs)
    return out.drop(columns=["_kx", "_ky"])


# >> Cumulative layer: every location with onset at or before `period`.
# >> E.g. cumulative_through(layer, 2022) = all mines first confirmed 2018..2022.
def cumulative_through(layer: gpd.GeoDataFrame, period: Period) -> gpd.GeoDataFrame:
    """Everything with an onset at or before ``period``."""
    order = {t: Period.parse(t).sort_key for t in layer["onset"].unique()}
    keep = layer["onset"].map(order) <= period.sort_key
    return layer.loc[keep].reset_index(drop=True)


# >> Chooses which periods get their own published cumulative layer.
# >> Resolvable annual periods, then (for the recent edge) quarters where they exist,
# >> else that year's annual. An annual year covered by quarters is only a witness.
def published_periods(periods: Sequence[Period],
                      config: PersistenceConfig) -> List[Period]:
    """The periods that become published layers, in order.

    Annual periods the rule can resolve publish as themselves. Past the last of
    those, the provisional edge publishes at the finest cadence available:
    quarters where they exist, otherwise that year's annual.

    An annual period that quarters already cover is still loaded as a *witness*
    -- 2024 cannot confirm without the 2025 annual -- but is not published.
    Witness and published layer are different roles, and coverage past the last
    confirmed year should arrive progressively rather than in year-sized jumps.
    """
    resolvable = [p for p in resolvable_periods(periods, config)]
    last_year = max((p.year for p in resolvable), default=None)

    quartered = {p.year for p in periods if not p.is_annual}
    out = list(resolvable)
    for p in sorted(periods):
        if p.is_annual and (last_year is None or p.year <= last_year):
            continue                       # already published, or resolvable
        if p.is_annual and p.year in quartered:
            continue                       # superseded by its own quarters
        out.append(p)
    return sorted(set(out))


# >> Writes one cumulative chip layer per period (+ each period's new chips).
# >> Files: <stem>_cumulative<start>-<tag>.geojson and
# >> patch_diffs/<stem>_growth_<tag>.geojson,
# >> GeoJSON at 6 decimals. patch_diffs are the SAM2 prompts for quarterly masks.
# >> Returns (period, total rows, confirmed rows) per period for the log table.
def write_cumulative_series(layer: gpd.GeoDataFrame,
                            periods: Sequence[Period],
                            outdir: Path,
                            stem: str,
                            patch_diffs: bool = True,
                            precision: int = PostprocessConfig.coordinate_precision
                            ) -> List[Tuple[Period, int, int]]:
    """Write one cumulative patch layer per period.

    Returns ``(period, total, confirmed)`` per period. The cumulatives are
    redundant with ``layer`` -- each is a filter on ``onset`` -- but are what
    consumers actually load, and per-period files are what QGIS wants for review.

    ``patch_diffs`` writes each period's new patches to ``patch_diffs/``. On by
    default: these are the SAM2 prompt set for quarterly segmentation, so the
    quarterly masks cannot be built without them. Specifically:
    the quarter's detections minus everything already accumulated, which is what
    keeps SAM2 off the cloud-wrecked bulk of a quarterly mosaic. Note the
    threshold differs by cadence and that is deliberate -- annual increments come
    from the t0.43 frames, quarterly from ``t_prov`` -- because a quarterly mask
    is replaced when its year closes, never promoted, so there is nothing for a
    looser quarterly prompt to be promoted into. See the planning doc,
    "Quarterly masks: segment the diff, not the period".

    They are not part of the published product -- for patches the increment is
    just the rows carrying that onset, recoverable from any cumulative. The
    published increments are the dissolved polygons, which are genuinely not
    recoverable by filtering; see write_dissolved_series.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if patch_diffs:
        (outdir / "patch_diffs").mkdir(exist_ok=True)
    start = min(periods).year
    summary = []
    # >> Loop in published-period order; each file includes all earlier onsets.
    for period in periods:
        cumulative = cumulative_through(layer, period)
        if cumulative.empty:
            continue
        # >> File write: cumulative chips up to this period.
        cumulative.to_file(
            outdir / f"{stem}_cumulative{start}-{period.tag}.geojson",
            driver="GeoJSON", index=False, COORDINATE_PRECISION=precision)
        if patch_diffs:
            # >> File write: only the chips whose onset is this period (the growth).
            increment = layer.loc[layer["onset"] == period.tag].reset_index(drop=True)
            increment.to_file(
                outdir / "patch_diffs" / f"{stem}_growth_{period.tag}.geojson",
                driver="GeoJSON", index=False, COORDINATE_PRECISION=precision)
        summary.append((period, len(cumulative),
                        int((cumulative["status"] == "confirmed").sum())))
    return summary


# >> Area of each polygon in hectares, measured on the WGS84 ellipsoid.
# >> Uses pyproj.Geod.geometry_area_perimeter (works on lon/lat directly).
# >> abs() because the sign depends on ring winding order; / 1e4 converts m^2 -> ha.
def polygon_area_ha(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Geodesic area of each polygon, in hectares.

    Geodesic rather than the per-UTM-zone reprojection used in
    scripts/boundaries: exact on the ellipsoid at any longitude, with no zone
    bookkeeping to get wrong.
    """
    # >> Geodesic calculator on the WGS84 ellipsoid (the Earth shape GPS uses).
    geod = Geod(ellps="WGS84")
    return np.array([abs(geod.geometry_area_perimeter(g)[0]) / 1e4
                     for g in gdf.geometry])


# >> Writes dissolved (merged-polygon) cumulatives per period, plus yearly growth.
# >> Growth = this period's cumulative polygons MINUS the previous union (geometric
# >> difference), so growth pieces do not overlap earlier years.
# >> Files: <stem>_cumulative<start>-<tag>-dissolved.geojson and
# >> diffs/<stem>_growth_<tag>-dissolved.geojson. Growth pieces <= 11 ha are dropped.
def write_dissolved_series(layer: gpd.GeoDataFrame,
                           periods: Sequence[Period],
                           outdir: Path,
                           stem: str,
                           # >> 11.0 ha (~half a 23 ha chip):
                           # >> smaller growth slivers are dropped.
                           min_polygon_ha: float = 11.0,
                           # >> Dissolve buffer in DEGREES (1e-5 ~ 1.1 m).
                           buffer_deg: float = PostprocessConfig.dissolve_buffer_deg,
                           precision: int = PostprocessConfig.coordinate_precision
                           ) -> List[Tuple[Period, int, int]]:
    """Write dissolved cumulative polygons per period, and their increments.

    Returns ``(period, cumulative polygons, increment polygons)``.

    The increment is a **geometric difference** against the running union, not
    ``dissolve(patches with onset Y)``: patches overlap by half a width, so
    dissolving a year's own patches would overlap the previous year's polygons
    instead of partitioning. The difference draws growth-by-year correctly and
    still unions back to the cumulative.

    This is why the cumulative dissolves are computed even though the increments
    are what get published -- they are the necessary intermediate.

    Each increment is status-homogeneous, since without early confirmation an
    annual period is either wholly confirmable or wholly provisional; that is
    what lets a polygon carry one unambiguous ``onset_year`` and ``status``.

    Increments below ``min_polygon_ha`` are dropped. This leaks: a consumer
    unioning the published increments falls short of the true cumulative by the
    dropped fragments. Acceptable because the polygons are display-only -- anyone
    wanting an exact cumulative should dissolve the patch layer, which is
    authoritative and gives the right answer by construction. The cumulative
    dissolves here are *not* filtered, since they are intermediates for the
    difference and filtering them would compound the leak year on year.
    """
    outdir = Path(outdir)
    (outdir / "diffs").mkdir(parents=True, exist_ok=True)
    start = min(periods).year
    summary, previous = [], None
    for period in periods:
        cumulative = cumulative_through(layer, period)
        if cumulative.empty:
            continue
        # >> Buffered dissolve of all chips so far (mean confidence per polygon).
        dissolved = dissolve_patches(cumulative, buffer_deg=buffer_deg)
        # >> File write: dissolved cumulative (NOT area-filtered).
        dissolved.to_file(
            outdir / f"{stem}_cumulative{start}-{period.tag}-dissolved.geojson",
            driver="GeoJSON", index=False, COORDINATE_PRECISION=precision)

        increment = dissolved.copy()
        if previous is not None:
            # >> Growth = new polygons minus everything that existed before.
            increment.geometry = dissolved.geometry.difference(previous)
            increment = increment.loc[~increment.geometry.is_empty]
            increment = increment.explode(index_parts=False).reset_index(drop=True)

        # >> Increment status: confirmed or provisional ('mixed' only if early_confirm).
        statuses = set(layer.loc[layer["onset"] == period.tag, "status"])
        increment["onset"] = period.tag
        increment["onset_year"] = period.year
        increment["status"] = statuses.pop() if len(statuses) == 1 else "mixed"

        if min_polygon_ha and len(increment):
            # >> Geodesic area per growth piece, hectares.
            increment["area_ha"] = polygon_area_ha(increment)
            keep = increment["area_ha"] > min_polygon_ha
            dropped = int((~keep).sum())
            if dropped:
                print(f"  {period.tag}: dropped {dropped:,} increment polygons "
                      f"<= {min_polygon_ha:g} ha "
                      f"({increment.loc[~keep, 'area_ha'].sum():,.0f} ha)")
            increment = increment.loc[keep].reset_index(drop=True)

        # >> File write: growth polygons for this period.
        increment.to_file(
            outdir / "diffs" / f"{stem}_growth_{period.tag}-dissolved.geojson",
            driver="GeoJSON", index=False, COORDINATE_PRECISION=precision)

        # >> Remember the union so the next period's growth excludes it.
        previous = dissolved.geometry.union_all()
        summary.append((period, len(dissolved), len(increment)))
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# >> CLI entry: load all periods, build the first-detection layer, write the series.
# >> Reads postprocessed files under --base; writes <base>/cumulative/ and
# >> <base>/cumulative_dissolved/ (with --dissolve, as pipeline.py passes).
def main(args: argparse.Namespace) -> None:
    config = PersistenceConfig(
        k=args.k, window=args.window,
        use_quarterly_witnesses=(args.witnesses == "all"),
        early_confirm=args.early_confirm, nested=args.nested,
        t_prov=args.t_prov)

    base = Path(args.base)
    # >> Parse --years and --quarters into Period objects and sort them in time order.
    periods = [Period.parse(t) for t in list(args.years) + list(args.quarters)]
    periods.sort()

    frames, provisional = {}, {}
    resolvable = set(resolvable_periods(periods, config))
    # A period withheld from publication contributes no provisional rows either.
    # Loading them would defeat the withholding: their onset tag sorts inside the
    # published sequence, so every later cumulative would sweep them back in --
    # the 2025 annual adds 27,910 locations to Q126 that its own quarters are
    # supposed to be introducing progressively.
    publishable = set(published_periods(periods, config))
    # >> Loop: load every period (main t0.43 file + Andes), plus t_prov files when needed.
    for p in periods:
        path = detection_path(base, p, region_stem=args.region_stem,
                              t_main=args.t_main, t_iso=args.t_iso)
        if not path.is_file():
            raise SystemExit(f"missing detections for {p.tag}: {path}")
        supp = None if args.no_supplemental else supplemental_path(
            base, p, stem=args.supplemental_stem,
            threshold=args.supplemental_threshold, subdir=args.supplemental_dir)
        # >> Main frames: the t0.43 / 0.75 detections (the witnesses).
        frames[p] = load_period(path, config.centroid_decimals, supp)
        # >> Resolvable or withheld periods get no provisional layer.
        if p in resolvable or p not in publishable:
            continue
        # >> Provisional frames: the t_prov (0.55 / 0.8) file for this period.
        prov_path = detection_path(base, p, region_stem=args.region_stem,
                                   t_main=config.t_prov, t_iso=args.t_prov_iso)
        if prov_path.is_file():
            provisional[p] = load_period(prov_path, config.centroid_decimals, supp)
        else:
            print(f"  ! no t_prov file for {p.tag}, skipping provisional layer")

    print(f"{len(frames)} periods, {len(resolvable)} resolvable under "
          f"k={config.k} window={config.window} witnesses={args.witnesses}")

    # >> Build the master one-row-per-location layer.
    layer = build_first_detection_layer(frames, provisional, config)
    # Derived from --base rather than defaulted literally, so pointing at a
    # different model writes that model's cumulative, not this one's.
    out = Path(args.outdir) if args.outdir else base / "cumulative"
    out.mkdir(parents=True, exist_ok=True)

    # The published stem carries no threshold tags: consumers get one product,
    # and which thresholds produced it is recorded by the postprocessed_* folder
    # names archived alongside.
    stem = args.region_stem
    # >> Strip threshold tags from the stem, e.g. '..._0.40' -> product name without it.
    for suffix in (f"_{args.t_main:g}", "_0.40"):
        stem = stem.replace(suffix, "")

    # >> File write: the first-detection layer (confirmed + provisional), GeoJSON 6 dp.
    dest = out / f"{stem}_detections.geojson"
    layer.to_file(dest, driver="GeoJSON", index=False,
                  COORDINATE_PRECISION=PostprocessConfig.coordinate_precision)

    confirmed = int((layer["status"] == "confirmed").sum())
    print(f"\n  {confirmed:,} confirmed + {len(layer) - confirmed:,} provisional "
          f"= {len(layer):,} locations -> {dest}")

    if args.no_series:
        return

    published = published_periods(periods, config)
    withheld = [p.tag for p in periods if p not in published]
    if withheld:
        print(f"  witnesses not published as layers: {', '.join(withheld)}")

    print(f"\n{'period':>8} {'patches':>10} {'confirmed':>11} {'provisional':>12}")
    # >> Write chip cumulatives and print the per-period table.
    for period, total, conf in write_cumulative_series(
            layer, published, out, stem, patch_diffs=args.patch_diffs):
        print(f"{period.tag:>8} {total:>10,} {conf:>11,} {total - conf:>12,}")
    print(f"\n  series -> {out}")

    # >> Optional polygons: writes to a sibling folder <outdir>_dissolved.
    if args.dissolve:
        dissolved_out = out.with_name(f"{out.name}_dissolved")
        print(f"\n{'period':>8} {'polygons':>10} {'increment':>11}")
        for period, npoly, ninc in write_dissolved_series(
                layer, published, dissolved_out, stem,
                min_polygon_ha=args.min_polygon_ha):
            print(f"{period.tag:>8} {npoly:>10,} {ninc:>11,}")
        print(f"\n  dissolved series -> {dissolved_out}")


if __name__ == "__main__":
    # >> Command-line options. Defaults = Recipe A and the LOOSE postprocess pair.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = PersistenceConfig()
    parser.add_argument("--base", default=str(BASE),
                        help="Model output directory holding postprocessed_*/ folders")
    parser.add_argument("--region_stem",
                        default=f"Amazon_ACA_{MODEL}_{RAW_TAG}")
    parser.add_argument("--years", nargs="*", default=[],
                        help="Annual periods, e.g. --years 2018 2019 2020")
    parser.add_argument("--quarters", nargs="*", default=[],
                        help="Quarterly periods, e.g. --quarters Q125 Q225")
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--window", type=int, default=defaults.window,
                        help="Window length in years, including the onset year")
    parser.add_argument("--witnesses", choices=("annual", "all"), default="annual",
                        help="'all' lets quarterly periods corroborate")
    parser.add_argument("--early-confirm", action="store_true",
                        help="Confirm without waiting for the window to close")
    parser.add_argument("--nested", action="store_true",
                        help="Also evaluate shorter windows; take the earliest onset")
    parser.add_argument("--t-main", dest="t_main", type=float,
                        default=PostprocessConfig.t_main)
    parser.add_argument("--t-iso", dest="t_iso", type=float,
                        default=PostprocessConfig.t_iso)
    parser.add_argument("--t-prov", dest="t_prov", type=float, default=defaults.t_prov)
    # >> t_prov_iso 0.8 = the isolated-chip bar of the provisional (stringent) files.
    parser.add_argument("--t-prov-iso", dest="t_prov_iso", type=float, default=0.8)
    parser.add_argument("--outdir", default=None,
                        help=("Published product directory; defaults to "
                              "<base>/cumulative, alongside raw_detections"))
    parser.add_argument("--supplemental_dir", default="raw_detections/andes_supplemental",
                        help="Supplemental detections, relative to --base ('' to disable)")
    parser.add_argument("--supplemental_stem",
                        default=f"andes_supplemental_{MODEL}")
    parser.add_argument("--supplemental_threshold", default=ANDES_TAG,
                        help="Raw threshold in the supplemental filenames")
    parser.add_argument("--no-supplemental", action="store_true",
                        help="Skip the Andes supplemental union")
    parser.add_argument("--no-series", action="store_true",
                        help="Write only the first-detection layer, not the per-period series")
    parser.add_argument("--no-patch-diffs", dest="patch_diffs",
                        action="store_false",
                        help="Skip patch_diffs/; they are the SAM2 quarterly prompt set")
    parser.add_argument("--dissolve", action="store_true",
                        help=("Also write dissolved cumulative polygons and their "
                              "geometric yearly increments to <outdir>_dissolved/"))
    # >> 11 ha minimum area for published growth polygons.
    parser.add_argument("--min-polygon-ha", dest="min_polygon_ha", type=float,
                        default=11.0,
                        help="Drop increment polygons at or below this area (0 disables)")
    main(parser.parse_args())
