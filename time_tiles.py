#!/usr/bin/env python3
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
time_tiles.py — measure seconds-per-tile for the NEW model (CNN, embedding_strategy='none').

Splits a small region into tiles and, per tile, times the three cost components
separately so you can extrapolate compute + cost to any area / cadence:

  1. FETCH cold  — first pull from Google Earth Engine (no cache) + write cache
  2. FETCH warm  — second pull, served from the local image cache
  3. INFER       — normalize + chip (48px, stride 24) + CNN predict on one tile

Run it AFTER you've pulled core/ from the branch and authenticated Earth Engine.
Place this file in core/ (or run from repo root — it will find core/ automatically).

Example:
  export EE_PROJECT=your-gcp-project
  cd core
  python time_tiles.py \
      --model ../models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 \
      --region ../data/boundaries/test_region.geojson \
      --start_date 2024-01-01 --end_date 2024-03-31 \
      --n_tiles 10
"""
# >> FILE OVERVIEW
# >> Fork-original benchmarking script (RA timeseries project): times, per tile, a
# >> cold EE download, a warm cache read and one CNN inference, then extrapolates
# >> to the whole basin at quarterly/monthly cadence to estimate compute cost.
# >> Uses upstream core/ unchanged (gee.py, tile_utils.py, inference_engine.py).
# >> Not part of the production pipeline; writes only image-cache GeoTIFFs.
import argparse, os, sys, time, statistics, tempfile
from pathlib import Path

# --- make the repo's core/ modules importable whether run from core/ or repo root ---
# >> Find core/ (this folder, ./core or ../core) and put it first on sys.path so
# >> the bare imports below (gee, tile_utils, inference_engine) work.
_here = Path(__file__).resolve().parent
for cand in (_here, _here / "core", _here.parent / "core"):
    if (cand / "gee.py").exists():
        sys.path.insert(0, str(cand)); break

import geopandas as gpd
import gee
import tile_utils
from inference_engine import InferenceConfig, InferenceEngine


# >> Rebuild gee.get_tile_data's cache filename so we can delete it (force cold).
# >> Must match gee.py exactly: {collection}_{tile.key}_{start}_{end}.tif.
def cache_file_for(cache_dir: Path, collection: str, tile, start: str, end: str) -> Path:
    return cache_dir / f"{collection}_{tile.key}_{start}_{end}.tif"


# >> Benchmark: parse args, tile the region, time N tiles, print stats + estimates.
# >> Output is printed only; no results file is written.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the CNN ensemble .h5")
    ap.add_argument("--region", required=True, help="Small AOI geojson (e.g. test_region.geojson)")
    ap.add_argument("--start_date", default="2024-01-01")
    # >> Gotcha: EE end date is EXCLUSIVE, so the default 2024-03-31 skips Mar 31.
    ap.add_argument("--end_date", default="2024-03-31")
    ap.add_argument("--n_tiles", type=int, default=10, help="How many tiles to time")
    ap.add_argument("--cache_dir", default=None, help="Image cache dir (default: temp)")
    # >> Tile count used for extrapolation (219,800 DLTiles for the whole basin).
    ap.add_argument("--basin_tiles", type=int, default=219800,
                    help="Tiles for full amazon_basin (for extrapolation)")
    args = ap.parse_args()

    # >> Cache folder: --cache_dir, or a fresh temp folder (amw_time_*) if not given.
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="amw_time_"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # >> Production data settings (552 + 12 pad, S2L1C, 0.6) with the cache switched on.
    data_config = gee.DataConfig(image_cache_dir=str(cache_dir))
    # >> embedding_strategy='none' = production CNN on raw pixels (no foundation model).
    infer_config = InferenceConfig(model_path=args.model, embedding_strategy="none")

    region = gpd.read_file(args.region)
    try:
        # >> Merge all AOI features into one geometry (union_all on new geopandas, else
        # >> the older unary_union).
        geom = region.union_all()
    except AttributeError:
        geom = region.unary_union

    # >> Region -> DLTiles; keep only the first n_tiles in grid order (not random).
    tiles = tile_utils.create_tiles(geom, data_config.tilesize, data_config.pad)
    tiles = tiles[: args.n_tiles]
    print(f"Region {args.region}: timing {len(tiles)} tile(s), window {args.start_date}..{args.end_date}")
    print(f"Image cache: {cache_dir}\n")

    # >> Loads the .h5 ensemble and creates the EE extractor (ee.Initialize runs here).
    # >> SAM2 is off (run_sam2 default False), so inference = CNN only.
    engine = InferenceEngine(args.start_date, args.end_date, data_config, infer_config)
    engine._ensure_keras_model()

    cold, warm, infer = [], [], []
    warmed_up = False

    # >> LOOP: one tile at a time (single process, no thread pool), so timings are
    # >> serial latencies, not the throughput of bulk_predict's 8 producer threads.
    for i, tile in enumerate(tiles):
        try:
            cf = cache_file_for(cache_dir, data_config.collection, tile, args.start_date, args.end_date)
            # >> Gotcha: this deletes that tile's cached image, even in a user-given
            # >> cache_dir.
            if cf.exists():
                cf.unlink()  # force a true cold fetch

            # >> Cold = EE computePixels download + reshape + writing the uint16 cache
            # >> GeoTIFF.
            t0 = time.perf_counter()
            pixels = engine.data_extractor.get_tile_data(tile)
            t_cold = time.perf_counter() - t0

            # >> Warm = read the cached GeoTIFF back (no network). Note: returns the
            # >> uint16 cached values, while inference below uses the float32 cold pixels.
            t0 = time.perf_counter()
            _ = engine.data_extractor.get_tile_data(tile)  # cache hit
            t_warm = time.perf_counter() - t0

            # >> Warm-up: first model.predict builds the TF graph; run once untimed.
            if not warmed_up:  # first predict pays TF graph-build cost; don't count it
                engine.predict_on_tile_pixels(pixels, tile)
                warmed_up = True

            # >> Infer = normalise /10000 + cut 529 chips + model.predict + threshold 0.5.
            t0 = time.perf_counter()
            engine.predict_on_tile_pixels(pixels, tile)
            t_infer = time.perf_counter() - t0

            cold.append(t_cold); warm.append(t_warm); infer.append(t_infer)
            print(f"  tile {i+1:>3}/{len(tiles)}  cold_fetch={t_cold:6.2f}s  "
                  f"warm_fetch={t_warm:6.3f}s  infer={t_infer:6.2f}s")
        except Exception as e:
            print(f"  tile {i+1}: ERROR {e}")

    if not cold:
        print("\nNo tiles timed successfully."); return

    # >> (mean, median) of a list of seconds.
    def stat(x): return (statistics.mean(x), statistics.median(x))
    mc, medc = stat(cold); mw, medw = stat(warm); mi, medi = stat(infer)

    print("\n===== per-tile seconds (single process) =====")
    print(f"  cold fetch : mean {mc:6.2f}  median {medc:6.2f}")
    print(f"  warm fetch : mean {mw:6.3f}  median {medw:6.3f}")
    print(f"  inference  : mean {mi:6.2f}  median {medi:6.2f}")

    # >> Per-tile totals use MEDIANS (robust to a few slow EE responses).
    # Warm steady-state per tile = warm fetch + inference (imagery already cached)
    warm_tile = medw + medi
    cold_tile = medc + medi
    print(f"\n  warm total/tile (warm fetch + infer) = {warm_tile:6.2f}s")
    print(f"  cold total/tile (cold fetch + infer) = {cold_tile:6.2f}s")

    # >> Extrapolation: tiles x periods per year x seconds per tile / 3600 = hours.
    # >> Gotcha: labelled 'core-hrs' but this is single-process wall-clock time; TF
    # >> and the EE download already use several threads/cores inside one process.
    # --- extrapolation to full basin, SINGLE core (scale down by your parallel procs) ---
    print("\n===== single-process extrapolation to full amazon_basin =====")
    print(f"  (basin_tiles = {args.basin_tiles:,}; divide wall-clock by your #parallel processes)")
    # >> Cadences: 4 quarterly or 12 monthly composites per year.
    for label, n_periods in (("quarterly/yr", 4), ("monthly/yr", 12)):
        jobs = args.basin_tiles * n_periods
        # cold once per composite (each period fetched once), then inference dominates re-runs
        hrs_warm = jobs * warm_tile / 3600
        hrs_cold = jobs * cold_tile / 3600
        print(f"  {label:14s} {jobs:>12,} tile-jobs | "
              f"~{hrs_cold:8.0f} core-hrs (cold) | ~{hrs_warm:8.0f} core-hrs (warm)")
    # >> Rough cost formula printed for the reader; the $1.55/hr price is hard-coded.
    print("\n  cost ≈ core-hours / vCPU_used * $/hr   (n2-standard-32 = 32 vCPU, ~$1.55/hr)")
    print("  e.g. 32 parallel procs on one box → machine-hours = core-hours / 32")


# >> Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
