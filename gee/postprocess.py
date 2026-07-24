#!/usr/bin/env python3
"""
postprocess.py — buffered dissolve, NDVI masking, and confidence filtering

=============================== FILE OVERVIEW ===============================
Post-inference cleanup for Amazon Mining Watch (AMW) mine detections.

Position in the pipeline:
  1. Upstream, the inference pipeline pulls Sentinel-2 tiles (576x576 px at
     10 m/px, 13 bands) from Google Earth Engine and cuts them into
     48x48-px chips (~480 m squares, ~23 ha each).
  2. A frozen SSL4EO ViT-S/16 backbone embeds each chip as a 384-d vector.
  3. A Keras ensemble of six MLP heads (384->64->16->1 sigmoid each) scores
     each embedding; the MEAN of the six sigmoid outputs is stored as the
     "confidence" property. Detections are written as chip squares (polygon
     features in EPSG:4326) into per-region GeoJSON files.
  4. THIS SCRIPT runs AFTER inference: it concatenates those GeoJSONs,
     thresholds on confidence, optionally dissolves touching chips into
     contiguous mine polygons, and drops small low-confidence polygons.

Typical invocation (run from the gee/ directory so `import gee` resolves):
    python postprocess.py <inference geojsons> --threshold 0.925 \
        --low_area_conf_threshold 0.975 --outpath out.geojson

Inputs on disk : one or more GeoJSON files of chip detections; features
                 carry a float "confidence" property in [0, 1].
Outputs on disk: a single GeoJSON at --outpath with the surviving polygons.

KNOWN ISSUES (also flagged inline below):
  * DEAD CODE: the --ndvi_threshold branch calls gee.Masker, a class that
    NO LONGER EXISTS in gee.py (NDVI masking was removed upstream; SAM2
    replaced it). Using --ndvi_threshold raises AttributeError. Do not use.
  * AREA CAVEAT: 'Polygon area (ha)' is computed in EPSG:3857 (Web
    Mercator), which inflates areas by ~sec^2(latitude). Near the equator
    the error is small (<1% within +/-5 deg latitude) but systematic. An
    equal-area CRS (e.g. EPSG:6933) would be correct.
"""

import argparse                 # CLI parsing for the __main__ entry point
from dataclasses import fields  # introspect gee.DataConfig fields to rebuild a config from CLI
import warnings                 # silence geopandas UserWarnings during degree-space buffering
from pathlib import Path        # portable output-path handling (mkdir of parent dirs)
import re                       # regex for the YYYY-MM-DD date validator below

import geopandas as gpd         # GeoDataFrames: buffer, union_all, explode, sjoin, file I/O
import pandas as pd             # pd.concat to stack multiple GeoDataFrames row-wise

import gee                      # local module: DataConfig etc. (Masker was REMOVED; see below)

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged.

    Used as an argparse ``type=`` callback for --start_date / --end_date,
    which are only consumed by the (dead) NDVI-masking branch that would
    need a GEE image date range.

    Inputs:
        s (str): candidate date string, e.g. "2023-01-01".
    Returns:
        str: ``s`` unchanged when it matches the YYYY-MM-DD shape.
    Raises:
        argparse.ArgumentTypeError: on mismatch, so argparse prints a clean
        usage error instead of a traceback.
    Side effects: none. Role in pipeline: CLI input validation only.
    """
    # Anchored regex: 4 digits '-' 2 digits '-' 2 digits. Checks shape only,
    # not calendar validity (e.g. "2023-99-99" would still pass).
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s  # valid shape: return the raw string for argparse to store
    # Wrong shape: argparse-specific error type yields a friendly CLI message.
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def dissolve_patches(
    gdf: gpd.GeoDataFrame, buffer_deg: float = 0.00001,
    conf_field: str = "confidence") -> gpd.GeoDataFrame:
    """Buffered dissolve of detection patches; aggregates confidence as mean.

    Merges per-chip square detections (48x48 px = ~480 m squares stored in
    EPSG:4326) into contiguous mine polygons: chips that touch or overlap
    become a single polygon. Chip corners may not align exactly in floating
    point, so each chip is first inflated by ``buffer_deg`` so that
    near-touching neighbors overlap; the union is deflated afterwards.

    Inputs:
        gdf (gpd.GeoDataFrame): chip detections in a geographic CRS
            (EPSG:4326, coordinates in DEGREES), one row per chip, with a
            float ``conf_field`` column in [0, 1].
        buffer_deg (float): inflation half-width in DEGREES. Default 1e-5
            deg (~1 m at the equator) — enough to bridge floating-point
            gaps without visibly changing shapes.
        conf_field (str): name of the confidence column to aggregate.

    Returns:
        gpd.GeoDataFrame: one row per connected component, in the input
        CRS, with ``conf_field`` set to the MEAN confidence of all input
        chips intersecting that polygon. Row count <= len(gdf).

    Side effects: prints the before/after polygon counts; writes no files.
    Role in pipeline: converts chip-level model output into the mine-scale
    polygons used in published AMW datasets.
    """
    gdf = gdf.copy()  # defensive copy so the caller's GeoDataFrame is untouched
    
    # geopandas warns when buffering in a geographic CRS (degrees are not a
    # linear unit). That is intentional here, so suppress the UserWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Inflate every chip by ~1e-5 deg (~1 m) with mitred joins
        # (join_style=2 keeps square corners square), then union_all()
        # dissolves the whole layer into one (Multi)Polygon geometry.
        dissolved_geom = gdf.buffer(buffer_deg, join_style=2).union_all()
        # Wrap the single unioned geometry in a one-row GeoDataFrame,
        # carrying over the input CRS (EPSG:4326).
        dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf.crs)
        # explode() splits the MultiPolygon into one row per connected
        # polygon; reset_index gives a clean 0..N-1 RangeIndex.
        dissolved = dissolved.explode(index_parts=False).reset_index(drop=True)
        # Negative buffer of the same width undoes the inflation, restoring
        # the true merged footprint (to within float precision).
        dissolved.geometry = dissolved.buffer(-buffer_deg, join_style=2)

    # Spatial join: pair each original chip with the dissolved polygon(s) it
    # intersects; 'index_right' labels the dissolved polygon in each pair.
    joined = gpd.sjoin(gdf, dissolved, how="inner", predicate="intersects")
    # Group pairs by dissolved polygon and average the member-chip confidences.
    mean_conf = joined.groupby("index_right")[conf_field].mean()
    # reindex aligns means to the dissolved rows (NaN if a polygon somehow had
    # no intersecting chip — should not happen, since the chips built it).
    dissolved[conf_field] = mean_conf.reindex(dissolved.index)

    dissolved.set_crs(gdf.crs, inplace=True)  # re-assert CRS metadata (EPSG:4326)
    print(f'Dissolved {len(gdf)} to {len(dissolved)} polygons.')  # progress log
    return dissolved

def filter_small_polygons(
    gdf, low_area_size=30.0, low_area_conf_threshold=0.975,
    conf_field="confidence", area_field="Polygon area (ha)"):
    """Filter polygons by dynamic confidence threshold depending on size.

    Small detections need higher confidence to survive, because small false
    positives (one or two stray chips) are common; large contiguous
    detections are rarely spurious.

    Inputs:
        gdf (gpd.GeoDataFrame): polygons with ``conf_field`` (float, [0,1])
            and ``area_field`` (float, HECTARES) columns.
        low_area_size (float): area cutoff in HECTARES (default 30 ha,
            ~1.3x one 23-ha chip); polygons below it count as "small".
        low_area_conf_threshold (float): minimum confidence a small polygon
            must reach to be kept (default 0.975).
        conf_field (str): name of the confidence column.
        area_field (str): name of the area column. NOTE: upstream this is
            computed in EPSG:3857, overstating true area by ~sec^2(lat).

    Returns:
        gpd.GeoDataFrame: copy of ``gdf`` without small low-confidence
        polygons; large polygons pass regardless of confidence.

    Side effects: none. Role in pipeline: final precision-boosting filter,
    applied after dissolve and before writing the output GeoJSON.
    """
    # Boolean Series marking rows to DROP: small in area AND below the
    # stricter confidence bar. Large polygons never match this mask.
    mask = ((gdf[area_field] < low_area_size) &
                (gdf[conf_field] < low_area_conf_threshold))
    # Keep the complement (~mask); .copy() detaches from the original frame.
    return gdf.loc[~mask].copy()

def main(args):
    """Postprocess polygons: dissolve, NDVI mask, and confidence filtering.

    Orchestrates the postprocessing sequence on parsed CLI args:
    load + concatenate -> confidence threshold -> buffered dissolve ->
    (dead) NDVI mask -> small-polygon filter -> write GeoJSON.

    Inputs:
        args (argparse.Namespace): parsed CLI arguments. Fields used here:
            geojson_paths (list[str]), threshold (float|None),
            dissolve (bool), ndvi_threshold (float|None, DEAD branch),
            start_date / end_date (str|None, YYYY-MM-DD),
            config (gee.DataConfig), low_area_size (float, hectares),
            low_area_conf_threshold (float|None), outpath (str).
    Returns: None.
    Side effects: reads the input GeoJSONs, prints progress, writes the
        filtered GeoJSON to args.outpath (parent dirs created as needed).
    Role in pipeline: the post-inference entry point that produces the
        final detection layer.
    """
    # Load every input as a GeoDataFrame and force EPSG:4326 so all inputs
    # share one geographic CRS (chip squares are natively stored in 4326).
    gdfs = [gpd.read_file(p).to_crs("EPSG:4326") for p in args.geojson_paths]
    print(f"{sum(len(g) for g in gdfs)} polygons from {len(gdfs)} files.")  # load log
    # Row-wise concat into one frame; ignore_index gives a fresh 0..N-1
    # index; crs= re-attaches the CRS that pd.concat does not preserve.
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    # Optional global confidence gate (e.g. 0.925): drop chips whose mean
    # ensemble sigmoid score falls below the threshold, before dissolving.
    if args.threshold is not None:
        gdf = gdf[gdf["confidence"] >= args.threshold].copy()

    # Optional buffered dissolve: merge touching chips into mine polygons.
    if args.dissolve:
        gdf = dissolve_patches(gdf)

    # ------------------------------------------------------------------
    # DEAD CODE — DO NOT USE. gee.Masker no longer exists in gee.py (NDVI
    # masking was removed upstream; SAM2 segmentation replaced it), so
    # passing --ndvi_threshold raises AttributeError at the gee.Masker
    # call below. Kept for historical reference only.
    # ------------------------------------------------------------------
    if args.ndvi_threshold is not None:
        # Would open a GEE connection for the given date range and config...
        extractor = gee.GEE_Data_Extractor(
            args.start_date, 
            args.end_date,
            args.config
        )
        # ...then CRASH here: gee.Masker was deleted (AttributeError).
        masker = gee.Masker(extractor, ndvi_threshold=args.ndvi_threshold)
        gdf = masker.ndvi_mask_polygons(gdf)  # unreachable in practice

    # Optional size-aware confidence filter (see filter_small_polygons).
    if args.low_area_conf_threshold is not None:
        # Compute areas only if not already present from an earlier stage.
        if not 'Polygon area (ha)' in gdf.columns:
            # CAVEAT: area via EPSG:3857 (Web Mercator) is inflated by
            # ~sec^2(latitude); <1% within +/-5 deg of the equator, but
            # systematic. An equal-area CRS (e.g. EPSG:6933) would be
            # correct. Units: .area gives m^2; /1e4 converts to hectares.
            gdf['Polygon area (ha)'] = gdf.to_crs('epsg:3857').area / 1e4
        # Drop small polygons that fail the stricter confidence bar.
        gdf = filter_small_polygons(
            gdf, low_area_size=args.low_area_size,
            low_area_conf_threshold=args.low_area_conf_threshold)
    
    outpath = Path(args.outpath)  # normalize to a Path object
    outpath.parent.mkdir(parents=True, exist_ok=True)  # ensure output dir exists
    # Write the final layer as GeoJSON; index=False keeps the pandas index
    # out of the feature properties.
    gdf.to_file(outpath, driver="GeoJSON", index=False)
    print(f"Wrote {len(gdf)} polygons to {outpath}")  # completion log

if __name__ == "__main__":
    # CLI definition. Only geojson_paths and --outpath are required; every
    # postprocessing stage is opt-in via its own flag.
    parser = argparse.ArgumentParser(
        description=("Postprocess polygons: dissolve, NDVI mask, and "
            "confidence filtering."))

    # Required args
    parser.add_argument(
        "geojson_paths", nargs="+",
        help="Input GeoJSON files to concatenate.",
    )  # one or more inference GeoJSONs (chip squares with 'confidence')
    parser.add_argument(
        "--outpath", required=True,
        help="Output GeoJSON file path.",
    )  # destination for the postprocessed detection layer

    # Postprocessing options
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Overall confidence threshold applied before dissolving.",
    )  # e.g. 0.925 in the typical invocation; None disables the gate
    parser.add_argument(
        "--dissolve", action="store_true",
        help="Run buffered dissolve (dissolve_patches). Default is False.",
    )  # merge touching chips into contiguous mine polygons
    # NOTE: --ndvi_threshold feeds the DEAD gee.Masker branch in main();
    # supplying it raises AttributeError. Do not use.
    parser.add_argument(
        "--ndvi_threshold", type=float, default=None,
        help="NDVI threshold for masking (omit to skip NDVI masking).",
    )
    parser.add_argument(
        "--start_date", type=valid_date,
        help="Start date in YYYY-MM-DD format")  # only used by dead NDVI branch
    parser.add_argument(
        "--end_date", type=valid_date,
        help="End date in YYYY-MM-DD format")  # only used by dead NDVI branch
    parser.add_argument("--low_area_size",
        type=float, default=30.0,
        help="Area (ha) below which to apply stricter confidence filtering.",
    )  # units: HECTARES; default 30 ha (~1.3 chips)
    parser.add_argument(
        "--low_area_conf_threshold", type=float, default=None,
        help="Confidence threshold for small polygons (< low_area_size).",
    )  # e.g. 0.975; None disables the small-polygon filter entirely

    # DataConfig args
    # These mirror gee.DataConfig fields so a config can be rebuilt below.
    # They only matter to the dead NDVI branch (which would query GEE);
    # for the normal dissolve/filter path they are inert.
    data_defaults = gee.DataConfig()  # dataclass instance holding the defaults

    parser.add_argument("--tilesize", type=int,
                        default=data_defaults.tilesize,
                        help="Tile width in pixels for requests to GEE")  # 576 px = 5.76 km
    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Number of pixels to pad each tile")  # default 0 px
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite image collection")  # e.g. S2L1C (13-band Sentinel-2)
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")  # in [0,1]; default 0.6
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Maximum concurrent GEE requests")  # default 8 threads
    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional directory to save/reload image rasters")  # None = no cache

    args = parser.parse_args()  # parse sys.argv into a Namespace

    # Guard: the (dead) NDVI branch would need a GEE date range, so require
    # both dates whenever --ndvi_threshold is given. parser.error() exits.
    if args.ndvi_threshold is not None:
        if args.start_date is None or args.end_date is None:
            parser.error("--start_date and --end_date are required "
                         "when --ndvi_threshold is provided.")
            
    # Rebuild a DataConfig from the CLI values: collect every DataConfig
    # field name and read the matching attribute off the parsed args.
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }
    # Restrict bands to the two NDVI bands (red=B4, nir=B8A) — again only
    # meaningful for the dead NDVI-masking path.
    config_dict.update({
        'bands': list(data_defaults._NDVI_BANDS.values())
    })
    
    args.config = gee.DataConfig(**config_dict)  # attach the built config to args
    main(args)  # run the postprocessing sequence

