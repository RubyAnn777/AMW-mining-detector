"""Which of our tiles has AMW ever found a mine in, and which tiles lie around them.

    python amw_comparison/history.py --run <run>
    sbatch cluster/amw_history.sbatch <run>                  the same, on Goethe

The question behind it: could we score only the tiles where AMW has found mining before
(plus a ring of neighbours), and still find nearly every mine? notebooks/amw_history.py
answers it for 2025 and 2026 with what this script writes.

Reads
    amw_comparison/data/amw/amazon_basin_detections.geojson   AMW's cumulative record
    <run>/plan/tiles.csv, <run>/plan/chips.parquet            our tiles and chips

Writes <run>/amw_comparison/
    history_tiles.parquet    one row per tile of the run: its key and number, its centre,
                             how many AMW history chips it holds, the first year AMW found
                             mining in it, and its ring for each history cut-off:
                             ring_to<Y> = 0 if AMW found mining in the tile by year Y,
                             1 if it touches such a tile, 2 one tile further, ...,
                             MAX_RING + 1 if further away than that.
    history_tiles.geojson    the same tiles' squares, those within MAX_RING of any AMW
                             history (for QGIS: the IDs and boundaries of the tiles)
    history_summary.json     how many AMW rows matched our chips, cut-offs, ring count

How a location becomes a tile: AMW's rows carry no tile, only each chip's square. We
compute AMW's location key from the square (centroid_key, as compare.py does) and look it
up in plan/chips.parquet, which holds the same key for every chip we plan. The chip's
chip_id then names its tile (chip_id // 1000). Where tiles overlap (UTM zone seams) a
location can be in two tiles; both count.

Rings: a tile's square is grown by half a tile less than k tiles on each side, in its own
UTM zone; every tile it overlaps is within ring k. Squares, not circles, so the eight
tiles around a tile are ring 1. Done with the squares rather than tile numbers so that
tiles across a UTM zone seam count as neighbours too.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402  (amw_comparison/compare.py: run folders, tile numbers)
from postprocess import centroid_key  # noqa: E402  (AMW code; compare.py put core/ on the path)

CUMULATIVE = compare.AMW_DIR / "amazon_basin_detections.geojson"
MAX_RING = 3                  # rings 0..3 are written; anything further is MAX_RING + 1
FAR = MAX_RING + 1


# ------------------------------------------------------------ keys and tiles

def key_int(x, y) -> np.ndarray:
    """AMW's location key (centre rounded to 5 decimals) as one whole number, for fast lookups."""
    kx = np.round((np.asarray(x, dtype=np.float64) + 180.0) * 1e5).astype(np.int64)
    ky = np.round((np.asarray(y, dtype=np.float64) + 90.0) * 1e5).astype(np.int64)
    return kx * 20_000_000 + ky


def split_tile(number) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile number (chip_id // 1000) -> UTM zone, tile column, tile row. Undoes compare.key_number."""
    number = np.asarray(number, dtype=np.int64)
    rest, row = np.divmod(number, 10_000)
    zone, col = np.divmod(rest, 1000)
    return zone, col - 500, row - 5000


def tile_squares(numbers, size_m: float, grow_m: float = 0.0, steps: int = 4) -> np.ndarray:
    """Each tile's square (without its pad) in longitude/latitude, grown by grow_m metres a side.

    The square is drawn in the tile's own UTM zone, then each side is cut into `steps`
    pieces before converting, so the outline stays true in degrees.
    """
    from descarteslabs.core.common.dltile.utm import FALSE_EASTING, utm_to_lonlat
    zone, col, row = split_tile(numbers)
    t = np.linspace(0.0, 1.0, steps + 1)[:-1]
    # the outline walked anticlockwise as fractions of the square: bottom, right, top, left
    fx = np.concatenate([t, np.ones_like(t), 1 - t, np.zeros_like(t)])
    fy = np.concatenate([np.zeros_like(t), t, np.ones_like(t), 1 - t])
    out = np.empty(len(zone), dtype=object)
    for z in np.unique(zone):
        m = zone == z
        x0 = col[m] * size_m + FALSE_EASTING - grow_m
        y0 = row[m] * size_m - grow_m
        side = size_m + 2 * grow_m
        xs = x0[:, None] + fx[None, :] * side
        ys = y0[:, None] + fy[None, :] * side
        lonlat = utm_to_lonlat(np.column_stack([xs.ravel(), ys.ravel()]), zone=int(z))
        out[m] = shapely.polygons(lonlat.reshape(m.sum(), len(fx), 2))
    return out


# ------------------------------------------------------------ reading

def read_cumulative(path: Path = CUMULATIVE) -> pd.DataFrame:
    """AMW's cumulative record: one row per location AMW ever found mining at, with the year."""
    if not path.exists():
        raise SystemExit(f"HISTORY ERROR: {path} is missing. Download amazon_basin_detections.geojson "
                         f"from https://source.coop/earthgenome/amazon-mining-watch (top folder) into "
                         f"{path.parent}")
    gdf = gpd.read_file(path, columns=["onset", "onset_year", "status"])
    kx, ky = centroid_key(gdf)
    return pd.DataFrame({"key": key_int(kx, ky), "onset": gdf["onset"].astype(str),
                         "onset_year": gdf["onset_year"].astype(int), "status": gdf["status"].astype(str)})


def chips_at(run: Path, keys: np.ndarray) -> pd.DataFrame:
    """Every planned chip whose AMW key is in `keys`: chip_id and key. Reads the plan in pieces."""
    import pyarrow.dataset as ds
    wanted = np.unique(keys)
    found = []
    scanner = ds.dataset(run / "plan" / "chips.parquet").scanner(
        columns=["chip_id", "amw_key_x", "amw_key_y"], batch_size=2_000_000)
    for i, batch in enumerate(scanner.to_batches()):
        k = key_int(batch.column("amw_key_x").to_numpy(), batch.column("amw_key_y").to_numpy())
        pos = np.searchsorted(wanted, k).clip(max=len(wanted) - 1)
        hit = wanted[pos] == k
        if hit.any():
            found.append(pd.DataFrame({"chip_id": batch.column("chip_id").to_numpy()[hit], "key": k[hit]}))
        if i % 10 == 0:
            print(f"  plan: {(i + 1) * 2:,} M chips read", flush=True)
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["chip_id", "key"])


# ------------------------------------------------------------ rings

def rings(tiles: pd.DataFrame, centre: np.ndarray, size_m: float, max_ring: int = MAX_RING) -> np.ndarray:
    """Per tile of `tiles`: 0 where centre is True, else the ring it is in around them (FAR beyond)."""
    out = np.where(centre, 0, max_ring + 1).astype(np.int8)
    if not centre.any():
        return out
    squares = tile_squares(tiles["tile"].to_numpy(), size_m)
    tree = shapely.STRtree(squares)
    centres = tiles.loc[centre, "tile"].to_numpy()
    for k in range(max_ring, 0, -1):          # widest first, so nearer rings overwrite
        grown = tile_squares(centres, size_m, grow_m=(k - 0.5) * size_m)
        _, hit = tree.query(grown, predicate="intersects")
        hit = np.unique(hit)
        out[hit] = np.minimum(out[hit], k)
    return out


# ------------------------------------------------------------ a run

def history_tiles(run: Path, cumulative: Path = CUMULATIVE) -> Path:
    t0 = time.time()
    out = run / "amw_comparison"
    out.mkdir(exist_ok=True)
    cfg = compare.frozen_config(run)["imagery"]
    size_m = float(cfg["tilesize"]) * 10.0           # grid.RESOLUTION_M: tiles are 10 m pixels

    amw = read_cumulative(cumulative)
    print(f"AMW cumulative: {len(amw):,} locations, onset {amw['onset_year'].min()}-"
          f"{amw['onset_year'].max()} ({time.time() - t0:.0f} s)", flush=True)

    tiles = pd.read_csv(run / "plan" / "tiles.csv").drop_duplicates("tile_key")
    tiles = pd.DataFrame({"tile": [compare.key_number(k) for k in tiles["tile_key"]],
                          "tile_key": tiles["tile_key"].to_numpy()})
    subs = pd.read_csv(run / "plan" / "tiles.csv")
    subs["tile"] = [compare.key_number(k) for k in subs["tile_key"]]
    tiles = tiles.merge(subs.groupby("tile")["subregion"].agg(lambda s: " ".join(map(str, sorted(s)))),
                        on="tile", how="left").sort_values("tile", ignore_index=True)

    chips = chips_at(run, amw["key"].to_numpy())
    chips["tile"] = chips["chip_id"].to_numpy() // 1000
    hits = chips.merge(amw, on="key")          # a location in two overlapping tiles: two rows
    matched = amw["key"].isin(chips["key"])
    print(f"matched {matched.sum():,} of {len(amw):,} AMW locations to our chips "
          f"({time.time() - t0:.0f} s)", flush=True)

    per_tile = hits.groupby("tile").agg(history_chips=("key", "nunique"), first_onset_year=("onset_year", "min"))
    tiles = tiles.merge(per_tile, on="tile", how="left")
    tiles["history_chips"] = tiles["history_chips"].fillna(0).astype(int)
    first = tiles["first_onset_year"]
    centre = shapely.centroid(tile_squares(tiles["tile"].to_numpy(), size_m, steps=1))
    tiles["lon"], tiles["lat"] = shapely.get_x(centre), shapely.get_y(centre)

    cutoffs = list(range(int(amw["onset_year"].min()), int(amw["onset_year"].max()) + 1))
    for year in cutoffs:
        tiles[f"ring_to{year}"] = rings(tiles, (first <= year).to_numpy(), size_m)
        n = tiles[f"ring_to{year}"].value_counts().sort_index()
        print(f"  history to {year}: " + ", ".join(f"ring {r}: {c:,}" for r, c in n.items() if r <= MAX_RING)
              + f" of {len(tiles):,} tiles", flush=True)
    tiles.to_parquet(out / "history_tiles.parquet", index=False)

    last = f"ring_to{cutoffs[-1]}"
    near = tiles[tiles[last] <= MAX_RING]
    gpd.GeoDataFrame(near.drop(columns=["lon", "lat"]),
                     geometry=tile_squares(near["tile"].to_numpy(), size_m), crs="EPSG:4326"
                     ).to_file(out / "history_tiles.geojson", driver="GeoJSON")

    summary = {
        "cumulative_file": cumulative.name, "cumulative_sha256": compare.sha256(cumulative)[:12],
        "amw_locations": int(len(amw)), "matched_to_our_chips": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "unmatched_by_onset_year": {int(k): int(v) for k, v in amw.loc[~matched, "onset_year"].value_counts().sort_index().items()},
        "locations_by_onset_year": {int(k): int(v) for k, v in amw["onset_year"].value_counts().sort_index().items()},
        "tiles_in_run": int(len(tiles)), "tile_size_m": size_m, "max_ring": MAX_RING, "cutoffs": cutoffs,
    }
    (out / "history_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results -> {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Our tiles in which AMW ever found mining, and their neighbours.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--cumulative", type=Path, default=CUMULATIVE, help=f"default: {CUMULATIVE}")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    history_tiles(run, args.cumulative)
