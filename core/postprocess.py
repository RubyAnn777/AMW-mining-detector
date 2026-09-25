#!/usr/bin/env python3
"""
postprocess.py — dual-threshold filtering of patch detections by spatial isolation.

Applies a main confidence threshold (t_main), computes each patch's distance to its
k-th nearest neighbor among surviving patches, then requires a stricter threshold
(t_iso) for isolated patches (distance > D km).
"""

# >> FILE OVERVIEW
# >> Cleans up one period's raw detections (chips scored > 0.4 by the CNN ensemble).
# >> Three tools: dedupe (seam duplicates), dual threshold (stricter bar for lonely
# >> chips), and buffered dissolve (merge touching chips into polygons).
# >> Pipeline: raw_detections/*.geojson -> THIS -> postprocessed_t*_d5_3km_t-iso*/ ->
# >> persistence.py. Called by scripts/pipeline.py:stage_postprocess (LOOSE + STRINGENT).
# >> Upstream Amazon Mining Watch file (byte-identical to upstream).
from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
# >> scikit-learn nearest-neighbour search; used with a KD-tree (a spatial index that
# >> finds nearby points fast without comparing every pair).
from sklearn.neighbors import NearestNeighbors

# >> Name of the column added to the output: distance (km) to the k-th nearest chip.
KTH_NEIGHBOR_FIELD = "kth_neighbor_km"


# >> Settings bundle for the whole detection catalogue (one place, so the CLI,
# >> persistence.py, concatenate.py and inference_engine.py cannot drift apart).
# >> @dataclass = Python auto-writes the constructor from the typed fields below.
@dataclass
class PostprocessConfig:
    """Detection-catalog conventions and the dual-threshold operating point.

    These are the parameters that define the published single-period products,
    so they live here rather than as argparse literals. The CLI defaults from
    an instance of this class.
    """

    # Dual-threshold operating point. t_main sits at the peak of the F0.5 curve;
    # isolated candidates (k-th nearest neighbour further than isolation_km)
    # must additionally clear t_iso. Website cumulatives have historically used
    # the stringent pair (0.55 / 0.8).
    # >> t_main 0.43 = main confidence bar (ensemble mean prob, 0-1) for every chip.
    # >> This is the LOOSE pair; pipeline.py also runs STRINGENT (0.55 / 0.8).
    t_main: float = 0.43
    # >> t_iso 0.75 = stricter bar a chip must ALSO clear if it is isolated.
    t_iso: float = 0.75
    # >> k = 5: isolation is judged by the distance to the 5th-nearest other chip.
    k: int = 5
    # >> isolation_km = 3.0 km: 5th neighbour further than this -> isolated.
    isolation_km: float = 3.0

    # >> Column holding the ensemble score (mean of the 6 CNN members' sigmoid outputs).
    confidence_field: str = "confidence"

    # GeoJSON coordinate precision. Left unset it floats with the GDAL/pyogrio
    # version on whichever machine ran the job -- inference years were run on
    # different VMs, and the 2024 vintage landed at 6 decimals while every other
    # year got full float64.
    #
    # 6 decimals is ~0.11 m, about 1/100 of a pixel. What makes it right is join
    # stability, not size. Cross-year joins key on a centroid rounded to
    # centroid_decimals; at full float64 the 2018-2025 union swings by 146,485
    # locations depending on whether the analyst keys at 4, 5, 6 or 7 decimals,
    # because sub-1e-5 jitter leaves the same ground location on either side of
    # a bin edge. Writing at 6 quantizes that away: the same union varies by
    # only 228, and 2023<->2024 matching rises from 126,138 to 132,353.
    #
    # This requires a UNIFORM archive. Mixing a newly written 6-dp file against
    # a legacy full-precision one costs 4.68% of matches (7 dp -0.53%, 8 dp
    # -0.06%, 9 dp -0.01%). If you write a file that must join against
    # un-rewritten legacy data, use 9.
    # >> GeoJSON written with 6 decimals of lat/lon (~0.11 m); pinned so joins are stable.
    coordinate_precision: int = 6

    # Join key for "same location, different period". The patch grid steps by
    # 0.00217 deg (half a patch), so 5 decimals resolves it ~217x while
    # absorbing sub-1e-5 jitter. 4 begins to collide.
    # >> Dedupe/join key = chip centre rounded to 5 decimals (~1.1 m). Chip centres step
    # >> 0.00217 deg (half a 48 px chip), so 5 dp never merges two real neighbours.
    centroid_decimals: int = 5

    # buffer_deg is in decimal degrees (~1 m at equator for 1e-5)
    # >> Dissolve buffer 1e-5 DEGREES (~1.1 m at the equator), not metres: just enough to
    # >> glue touching chips whose edges differ by float noise.
    dissolve_buffer_deg: float = 0.00001
    # >> Dissolve happens in EPSG:4326 = plain longitude/latitude on WGS84.
    dissolve_crs: str = "EPSG:4326"


# >> One default instance; the module-level names below are shortcuts into it.
_defaults = PostprocessConfig()

# Module-level aliases for the most widely imported values.
DEFAULT_CONFIDENCE_FIELD = _defaults.confidence_field
GEOJSON_COORDINATE_PRECISION = _defaults.coordinate_precision
CENTROID_DECIMALS = _defaults.centroid_decimals
DISSOLVE_CRS = _defaults.dissolve_crs


# >> Makes a per-chip location ID (rounded centre x, y) used to match the same chip
# >> across files and periods (dedupe here; onsets in persistence.py:load_period).
# >> In: GeoDataFrame of axis-aligned square chips in EPSG:4326. Out: two float64 arrays.
# >> Why round corners to 6 dp first: key what will be STORED, so dedupe is idempotent.
# >> Uses bounds midpoint (not .centroid) to avoid shapely's geographic-CRS warning.
def centroid_key(gdf, decimals=CENTROID_DECIMALS,
                 write_precision=GEOJSON_COORDINATE_PRECISION):
    """Rounded patch-centre key for joining a location across periods.

    Uses bounds midpoints: patches are axis-aligned boxes, and this avoids
    shapely's centroid-in-geographic-CRS warning.

    Corners are first rounded to ``write_precision`` so the key is computed on
    the coordinates as they will be *stored*. Without that, deduplication is
    not idempotent: writing at 6 decimals quantizes away the sub-1e-5 jitter
    between records, and pairs distinct under a 5-decimal key beforehand can
    collide afterwards. Pass ``write_precision=None`` to key on the input as
    given.
    """
    # >> Bounding box per chip: columns minx, miny, maxx, maxy (degrees).
    b = gdf.geometry.bounds
    minx, maxx = b.minx.to_numpy(), b.maxx.to_numpy()
    miny, maxy = b.miny.to_numpy(), b.maxy.to_numpy()
    if write_precision is not None:
        # >> Mimic the 6-decimal GeoJSON write before computing the key.
        minx, maxx, miny, maxy = (np.round(v, write_precision)
                                  for v in (minx, maxx, miny, maxy))
    # >> Centre = midpoint of the box, then rounded to 5 dp -> the location key.
    return (np.round((minx + maxx) / 2.0, decimals),
            np.round((miny + maxy) / 2.0, decimals))


# >> Removes duplicate chips, keeping the copy with the HIGHEST confidence.
# >> Why: the Amazon is scored as 6 subregion jobs; tiles on a subregion seam are
# >> scored by both jobs, so the same chip appears twice with slightly different scores.
# >> Keeping the max = what one run would have kept. Also merges Andes supplemental rows.
# >> Must run BEFORE dual_threshold_filter: a twin at 0 km fakes a close neighbour.
# >> Called by scripts/concatenate.py:concatenate and persistence.py:load_period.
def dedupe_detections(gdf, confidence_field=DEFAULT_CONFIDENCE_FIELD,
                      decimals=CENTROID_DECIMALS, verbose=True):
    """Drop repeated patch locations, keeping the highest-confidence record.

    Inference is run as six Amazon_ACA subregions; tiles straddling a subregion
    boundary are generated by both adjacent runs and emit the same detections
    twice. About half of duplicate pairs differ slightly in confidence (median
    delta 5e-4), the subregions having been scored in separate runs, so keeping
    the maximum matches what the threshold would have done had the location
    appeared once.

    Dedupe *before* postprocessing: a duplicate sits at distance zero from its
    twin, depressing the k-th-neighbour distance used to judge isolation, so a
    pair can slip past the t_iso cutoff that should have rejected it.
    """
    if len(gdf) == 0:
        return gdf

    # >> Location key per row (rounded centre).
    kx, ky = centroid_key(gdf, decimals)
    work = gdf.copy()
    work["_kx"], work["_ky"] = kx, ky
    # >> Sort best-first; mergesort is 'stable' (ties keep file order), so the result is
    # >> reproducible run to run.
    if confidence_field in work.columns:
        work = work.sort_values(confidence_field, ascending=False, kind="mergesort")

    # >> Keep the first (= highest-confidence) row for each location key.
    deduped = work.drop_duplicates(subset=["_kx", "_ky"], keep="first")
    # >> Restore original row order and drop the helper key columns.
    deduped = deduped.sort_index().drop(columns=["_kx", "_ky"])

    n_dropped = len(gdf) - len(deduped)
    if verbose and n_dropped:
        print(f"Dropped {n_dropped} duplicate detections "
              f"({n_dropped / len(gdf) * 100:.2f}%); {len(deduped)} remain.")
    return deduped


# >> Converts chip centres to metres so distances can be measured in km.
# >> In: GeoDataFrame (assumed EPSG:4326 if unset). Out: (n, 2) float64 array [x, y] in m.
# >> CRS conversion: estimate_utm_crs picks ONE UTM zone for the whole file (from its
# >> centre). Chips far from that zone get slightly stretched distances.
# >> Falls back to Web Mercator (EPSG:3857), which exaggerates distance away from equator.
def centroids_xy_m(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Patch centroids as (n, 2) coordinates in meters (projected CRS)."""
    g = gdf.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    try:
        # >> Reproject lon/lat -> metres (UTM = Transverse Mercator, 6-deg zones).
        g = g.to_crs(g.estimate_utm_crs())
    except Exception:
        g = g.to_crs("EPSG:3857")
    # >> Centroids are now in projected metres, so no geographic-CRS warning here.
    c = g.geometry.centroid
    return np.column_stack([c.x.to_numpy(dtype=np.float64), c.y.to_numpy(dtype=np.float64)])


# >> Measures how far each chip is from its k-th nearest OTHER chip, in km.
# >> In: (n, 2) coordinates in metres, k (5). Out: (n,) float64 km; all NaN if n <= k.
# >> Uses sklearn NearestNeighbors with a KD-tree. Asks for k+1 neighbours because
# >> each point's nearest neighbour is itself (distance 0) in column 0.
# >> Gotcha: NaN (too few chips in the file) is later treated as 'isolated'.
def kth_neighbor_km_on_catalog(coords_m: np.ndarray, k: int) -> np.ndarray:
    """Distance (km) from each point to its k-th nearest *other* point in coords_m."""
    n = len(coords_m)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= k:
        return out
    # >> k+1 neighbours: column 0 is the point itself, so column k is the k-th other chip.
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn.fit(coords_m)
    # >> dists_m shape (n, k+1), sorted nearest first, in metres.
    dists_m, _ = nn.kneighbors(coords_m)
    # >> Take column k and convert metres -> km.
    out[:] = dists_m[:, k] / 1000.0
    return out


# >> The dual-threshold decision rule, per chip -> True (keep) / False (drop).
# >> In: confidence (n,) and kth_neighbor_km (n,). Out: boolean mask (n,).
# >> Rule: every chip needs >= t_main; an ISOLATED chip (k-th neighbour > isolation_km,
# >> or distance NaN) must also reach >= t_iso. Why: real mines cluster; a lone
# >> high-ish score is more often a cloud edge or river bar.
def keep_dual_threshold(
    confidence: np.ndarray,
    kth_neighbor_km: np.ndarray,
    *,
    t_main: float,
    t_iso: float,
    isolation_km: float,
) -> np.ndarray:
    """
    Return boolean mask of patches to keep.

    Positive if confidence >= t_main; isolated patches (kth_neighbor_km > isolation_km,
    or non-finite distance) also require confidence >= t_iso.
    """
    # >> Isolated = neighbour distance known and larger than the cutoff (3 km)...
    is_isolated = np.isfinite(kth_neighbor_km) & (kth_neighbor_km > isolation_km)
    if np.any(~np.isfinite(kth_neighbor_km)):
        # >> ...or unknown (NaN, i.e. too few chips to measure): also treated as isolated.
        is_isolated = is_isolated | ~np.isfinite(kth_neighbor_km)
    above_main = confidence >= t_main
    above_iso = confidence >= t_iso
    # >> Isolated chips need both bars; clustered chips need only t_main.
    return np.where(is_isolated, above_main & above_iso, above_main)


# >> Formats a number compactly for filenames: 0.43 -> '0.43', 3.0 -> '3'.
def _fmt_param(value: float) -> str:
    return f"{value:g}"


# >> Builds the output filename, e.g. X_t0.43_d5_3km_t-iso0.75.geojson, beside input.
# >> pipeline.py:stage_postprocess then moves it into postprocessed_<tag>/.
def default_outpath(
    inpath: Path,
    *,
    t_main: float,
    k: int,
    isolation_km: float,
    t_iso: float,
) -> Path:
    """Output path: <stem>_t{t_main}_d{k}_{D}km_t-iso{t_iso}.geojson (notebook convention)."""
    stem = inpath.stem
    tag = (
        f"_t{_fmt_param(t_main)}_d{k}_{_fmt_param(isolation_km)}km"
        f"_t-iso{_fmt_param(t_iso)}"
    )
    return inpath.parent / f"{stem}{tag}.geojson"


# >> Adds '-dissolved' before the extension: a.geojson -> a-dissolved.geojson.
def dissolved_outpath(patches_outpath: Path) -> Path:
    """Patch output path with -dissolved before the extension."""
    return patches_outpath.with_name(f"{patches_outpath.stem}-dissolved{patches_outpath.suffix}")


# >> Merges touching/overlapping chip squares into single mine polygons.
# >> In: chip GeoDataFrame; buffer_deg in DEGREES (1e-5 ~ 1.1 m). Out: one row per
# >> polygon with the MEAN confidence of the chips it contains.
# >> Method ('buffered dissolve'): grow each square a little, union all, split into
# >> separate pieces, shrink back by the same amount.
# >> Called by main(--dissolve) and persistence.py:write_dissolved_series.
def dissolve_patches(
    gdf: gpd.GeoDataFrame,
    buffer_deg: float = 0.00001,
    conf_field: str = DEFAULT_CONFIDENCE_FIELD,
) -> gpd.GeoDataFrame:
    """
    Buffered dissolve of detection patches; aggregates confidence as mean.

    Reprojects to EPSG:4326 before buffering: ``buffer_deg`` is in decimal degrees
    (repo convention for inference GeoJSON).
    """
    gdf = gdf.copy()
    # >> CRS: force/convert to EPSG:4326 so buffer_deg really is degrees.
    if gdf.crs is None:
        gdf = gdf.set_crs(DISSOLVE_CRS)
    else:
        gdf = gdf.to_crs(DISSOLVE_CRS)

    # >> Silence shapely's 'buffer in a geographic CRS' warning (degrees are intended).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # >> Grow each square by buffer_deg (join_style=2 = mitred, keeps square corners),
        # >> then union_all merges everything into one big (Multi)Polygon.
        dissolved_geom = gdf.buffer(buffer_deg, join_style=2).union_all()
        dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf.crs)
        # >> explode: one MultiPolygon -> one row per connected piece.
        dissolved = dissolved.explode(index_parts=False).reset_index(drop=True)
        # >> Negative buffer shrinks back, undoing the growth.
        dissolved.geometry = dissolved.buffer(-buffer_deg, join_style=2)

    # >> Spatial join: which chips fall in which polygon (index_right = polygon row).
    joined = gpd.sjoin(gdf, dissolved, how="inner", predicate="intersects")
    # >> Average confidence of the chips inside each polygon.
    mean_conf = joined.groupby("index_right")[conf_field].mean()
    dissolved[conf_field] = mean_conf.reindex(dissolved.index)

    dissolved.set_crs(gdf.crs, inplace=True)
    print(f"Dissolved {len(gdf)} patches to {len(dissolved)} polygons.")
    return dissolved


# >> Applies the dual-threshold rule to a whole file of chips.
# >> In: GeoDataFrame with a 'confidence' column. Out: surviving rows plus a new
# >> column kth_neighbor_km. Called by main() and by pipeline via this CLI.
# >> Design: neighbours are counted only among chips >= t_main (the 'catalogue'),
# >> so weak detections below 0.43 cannot make a chip look clustered.
def dual_threshold_filter(
    gdf: gpd.GeoDataFrame,
    *,
    t_main: float,
    t_iso: float,
    k: int,
    isolation_km: float,
    confidence_field: str = DEFAULT_CONFIDENCE_FIELD,
) -> gpd.GeoDataFrame:
    """
    Filter detections with the dual-threshold rule.

    1. Keep patches with confidence >= t_main (neighbor catalog).
    2. Compute k-th nearest-neighbor distance on that catalog.
    3. Drop isolated catalog patches (distance > isolation_km) below t_iso.
    """
    if confidence_field not in gdf.columns:
        raise KeyError(
            f"Input needs {confidence_field!r}; columns: {list(gdf.columns)}"
        )

    # >> Scores as float64 array (n,).
    confidence = gdf[confidence_field].to_numpy(dtype=np.float64)
    # >> Step 1: the catalogue = chips at or above t_main.
    at_main = gdf.loc[confidence >= t_main].copy()
    if len(at_main) == 0:
        out = at_main.copy()
        out[KTH_NEIGHBOR_FIELD] = np.array([], dtype=np.float64)
        return out

    # >> Step 2: centres in metres (single UTM zone), then k-th neighbour distance.
    coords_m = centroids_xy_m(at_main)
    at_main[KTH_NEIGHBOR_FIELD] = kth_neighbor_km_on_catalog(coords_m, k=k)

    # >> Step 3: drop isolated catalogue chips that are below t_iso.
    keep = keep_dual_threshold(
        at_main[confidence_field].to_numpy(dtype=np.float64),
        at_main[KTH_NEIGHBOR_FIELD].to_numpy(dtype=np.float64),
        t_main=t_main,
        t_iso=t_iso,
        isolation_km=isolation_km,
    )
    return at_main.loc[keep].copy()


# >> CLI entry: read one GeoJSON, dual-threshold it, write the result (+ dissolve).
# >> Reads raw_detections/<region>_<start>_<end>.geojson; writes GeoJSON at 6 dp.
def main(args: argparse.Namespace) -> None:
    inpath = Path(args.geojson_path)
    # >> Read input detections (GeoJSON, EPSG:4326 lon/lat).
    gdf = gpd.read_file(inpath)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    n_in = len(gdf)
    filtered = dual_threshold_filter(
        gdf,
        t_main=args.t_main,
        t_iso=args.t_iso,
        k=args.k,
        isolation_km=args.D,
        confidence_field=args.confidence_field,
    )

    outpath = Path(args.outpath) if args.outpath else default_outpath(
        inpath,
        t_main=args.t_main,
        k=args.k,
        isolation_km=args.D,
        t_iso=args.t_iso,
    )
    # >> File write: filtered chips, GeoJSON, coordinates rounded to 6 decimals.
    outpath.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(outpath, driver="GeoJSON", index=False,
                     COORDINATE_PRECISION=GEOJSON_COORDINATE_PRECISION)

    # >> Stats for the log line only.
    n_at_main = int((gdf[args.confidence_field] >= args.t_main).sum())
    n_iso = int(
        (filtered[KTH_NEIGHBOR_FIELD] > args.D).sum()
        if len(filtered) and KTH_NEIGHBOR_FIELD in filtered.columns
        else 0
    )
    msg = (
        f"Read {n_in} patches from {inpath.name}; "
        f"{n_at_main} at t_main={args.t_main:g}; "
        f"wrote {len(filtered)} ({n_iso} isolated with d > {args.D:g} km) "
        f"to {outpath}"
    )
    # >> Optional (--dissolve): also write the merged-polygon layer. Note pipeline.py
    # >> does not pass --dissolve; persistence.py dissolves the cumulatives instead.
    if args.dissolve:
        dissolved_path = dissolved_outpath(outpath)
        dissolved = dissolve_patches(
            filtered, conf_field=args.confidence_field
        )
        # >> File write: dissolved polygons, GeoJSON, 6 decimals.
        dissolved.to_file(dissolved_path, driver="GeoJSON", index=False,
                          COORDINATE_PRECISION=GEOJSON_COORDINATE_PRECISION)
        msg += f"; dissolved -> {dissolved_path}"
    print(msg)


if __name__ == "__main__":
    # >> Command-line options; defaults come from PostprocessConfig above.
    defaults = PostprocessConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Dual-threshold post-processing: filter patch detections by confidence "
            "and spatial isolation (k-th nearest neighbor distance)."
        ),
    )
    parser.add_argument(
        "geojson_path",
        help="Input GeoJSON with patch detections and a confidence field.",
    )
    parser.add_argument(
        "--outpath",
        default=None,
        help=(
            "Output GeoJSON path. Default: input stem + "
            "_t{t_main}_d{k}_{D}km_t{t_iso}.geojson"
        ),
    )
    parser.add_argument(
        "--t-main",
        dest="t_main",
        type=float,
        default=defaults.t_main,
        help=f"Main confidence threshold (default: {defaults.t_main:g}).",
    )
    parser.add_argument(
        "--t-iso",
        dest="t_iso",
        type=float,
        default=defaults.t_iso,
        help=f"Stricter threshold for isolated patches (default: {defaults.t_iso:g}).",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=defaults.k,
        help=f"Neighbor rank for isolation distance, e.g. 5 = 5th NN (default: {defaults.k}).",
    )
    parser.add_argument(
        "-D",
        "--D",
        type=float,
        default=defaults.isolation_km,
        help="Isolation cutoff in km; patches with k-th NN distance > D use t_iso.",
    )
    parser.add_argument(
        "--confidence-field",
        default=defaults.confidence_field,
        help=f"Confidence column name (default: {defaults.confidence_field}).",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help=(
            "Also write a buffered-dissolve polygon layer as "
            "<patch-outstem>-dissolved.geojson"
        ),
    )

    main(parser.parse_args())
