"""Which of our tiles has AMW ever detected mining in, and which tiles lie around them.

    python amw_comparison/history.py --run <run>
    sbatch cluster/amw_history.sbatch <run>                  the same, on Goethe

The question behind it: score first only the tiles where AMW has ever detected mining,
plus rings of neighbours (priority 1), then the rest (priority 2). notebooks/amw_history.py
shows how many of our 2025/2026 detections priority 1 holds.

"Ever detected" = listed in any of AMW's raw detection files (confidence > 0.4, before any
postprocessing): amw_comparison/data/amw/amazon_basin_<period>_t0.4.geojson, every year and
quarter that is there. Not the cumulative record (amazon_basin_detections.geojson), which is
AMW's cleaned product (isolation thresholds, two-year confirmation), and not the Andes files.

Writes <run>/amw_comparison/
    history_chips.parquet    the unique chip IDs of every location AMW ever detected: chip_id,
                             tile, first_year, first_period, periods (how many files list it),
                             annual_periods
    history_tiles.parquet    one row per tile of the run: key and number, centre, country,
                             history_chips, first_year, ring (0 = AMW detected mining in the
                             tile, 1 = touches such a tile, ..., MAX_RING + 1 = further)
    history_tiles.geojson    the squares of the tiles within MAX_RING rings (for QGIS)
    history_summary.json     files read, how many AMW locations matched our chips

Location to chip: AMW's rows carry no chip ID, only each chip's square. We compute AMW's
location key from the square (centroid_key, as compare.py does) and look it up in
plan/chips.parquet; the chip_id then names its tile (chip_id // 1000). Where tiles overlap
(UTM zone seams) a location is in two tiles; both count.

Rings: a tile's square is grown by half a tile less than k tiles on each side, in its own
UTM zone; every tile it overlaps is within ring k. So the eight tiles around a tile are
ring 1, also across a UTM zone seam.
"""
from __future__ import annotations

import argparse
import json
import re
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

COUNTRIES = REPO / "data" / "boundaries" / "south_america_countries.geojson"   # Natural Earth 1:110m
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

RAW_NAME = re.compile(r"^amazon_basin_(\d{4}|Q[1-4]\d{2})_t0\.4\.geojson$")


def period_year(tag: str) -> int:
    """'2021' -> 2021, 'Q325' -> 2025."""
    return int(tag) if tag.isdigit() else 2000 + int(tag[2:])


def raw_files(folder: Path = compare.AMW_DIR) -> dict[str, Path]:
    """AMW's raw detection files in the folder, by period: {'2018': ..., 'Q125': ...}.

    Only the basin files at AMW's raw threshold (amazon_basin_<period>_t0.4.geojson):
    not the Andes files (threshold 0.2), not the postprocessed ones (_t0.43_t-iso0.75),
    not the cumulative record.
    """
    files = {m.group(1): p for p in folder.glob("amazon_basin_*.geojson") if (m := RAW_NAME.match(p.name))}
    if not files:
        raise SystemExit(f"HISTORY ERROR: no amazon_basin_<period>_t0.4.geojson in {folder}. Download "
                         f"them from https://source.coop/earthgenome/amazon-mining-watch (raw_detections/)")
    return dict(sorted(files.items(), key=lambda kv: (period_year(kv[0]), kv[0])))


def read_raw(files: dict[str, Path]) -> pd.DataFrame:
    """Every location any raw file lists (confidence > 0.4): key, first year, periods it is in."""
    parts = []
    for tag, path in files.items():
        gdf = gpd.read_file(path, columns=["confidence"])
        kx, ky = centroid_key(gdf)
        parts.append(pd.DataFrame({"key": key_int(kx, ky), "tag": tag, "year": period_year(tag)}))
        print(f"  {path.name}: {len(gdf):,} rows", flush=True)
    rows = pd.concat(parts, ignore_index=True).drop_duplicates(["key", "tag"])
    return (rows.groupby("key")
            .agg(first_year=("year", "min"), periods=("tag", "size"),
                 annual_periods=("tag", lambda t: int(t.str.isdigit().sum())),
                 first_period=("tag", "first"))
            .reset_index())


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
            print(f"  plan: batch {i + 1:,} read", flush=True)
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

def countries_of(lon, lat, path: Path = COUNTRIES) -> np.ndarray:
    """The country each point lies in (the nearest one for points just off the coarse outlines)."""
    if not path.exists():
        return np.full(len(lon), "", dtype=object)
    shapes = gpd.read_file(path)
    tree = shapely.STRtree(shapes.geometry.values)
    idx = tree.nearest(shapely.points(lon, lat))
    return shapes["country"].to_numpy()[idx]


def history_tiles(run: Path, folder: Path = compare.AMW_DIR) -> Path:
    t0 = time.time()
    out = run / "amw_comparison"
    out.mkdir(exist_ok=True)
    size_m = float(compare.frozen_config(run)["imagery"]["tilesize"]) * 10.0   # 10 m pixels

    files = raw_files(folder)
    print(f"AMW raw files: {' '.join(files)}", flush=True)
    amw = read_raw(files)
    print(f"{len(amw):,} locations in any of them ({time.time() - t0:.0f} s)", flush=True)

    listed = pd.read_csv(run / "plan" / "tiles.csv")
    listed["tile"] = [compare.key_number(k) for k in listed["tile_key"]]
    tiles = (listed.groupby("tile")
             .agg(tile_key=("tile_key", "first"),
                  subregion=("subregion", lambda s: " ".join(map(str, sorted(s)))))
             .reset_index())

    # the extra step: the unique chip IDs of every location AMW ever detected
    chips = chips_at(run, amw["key"].to_numpy())
    chips["tile"] = chips["chip_id"].to_numpy() // 1000
    hits = chips.merge(amw, on="key")          # a location in two overlapping tiles: two rows
    matched = amw["key"].isin(chips["key"])
    hits.drop(columns="key").sort_values("chip_id").to_parquet(out / "history_chips.parquet", index=False)
    print(f"matched {matched.sum():,} of {len(amw):,} AMW locations to {len(hits):,} of our chips "
          f"({time.time() - t0:.0f} s)", flush=True)

    per_tile = hits.groupby("tile").agg(history_chips=("chip_id", "size"), first_year=("first_year", "min"))
    tiles = tiles.merge(per_tile, on="tile", how="left")
    tiles["history_chips"] = tiles["history_chips"].fillna(0).astype(int)
    centre = shapely.centroid(tile_squares(tiles["tile"].to_numpy(), size_m, steps=1))
    tiles["lon"], tiles["lat"] = shapely.get_x(centre), shapely.get_y(centre)
    tiles["country"] = countries_of(tiles["lon"].to_numpy(), tiles["lat"].to_numpy())
    tiles["ring"] = rings(tiles, (tiles["history_chips"] > 0).to_numpy(), size_m)
    n = tiles["ring"].value_counts().sort_index()
    print("  " + ", ".join(f"ring {r}: {c:,}" for r, c in n.items() if r <= MAX_RING)
          + f" of {len(tiles):,} tiles", flush=True)
    tiles.to_parquet(out / "history_tiles.parquet", index=False)

    near = tiles[tiles["ring"] <= MAX_RING]
    gpd.GeoDataFrame(near.drop(columns=["lon", "lat"]),
                     geometry=tile_squares(near["tile"].to_numpy(), size_m), crs="EPSG:4326"
                     ).to_file(out / "history_tiles.geojson", driver="GeoJSON")

    summary = {
        "source": "AMW raw detections, confidence > 0.4", "files": {t: p.name for t, p in files.items()},
        "fingerprints": {t: compare.sha256(p)[:12] for t, p in files.items()},
        "amw_locations": int(len(amw)), "matched_to_our_chips": int(matched.sum()),
        "unmatched": int((~matched).sum()), "history_chips": int(len(hits)),
        "locations_by_first_year": {int(k): int(v) for k, v in amw["first_year"].value_counts().sort_index().items()},
        "tiles_in_run": int(len(tiles)), "tile_size_m": size_m, "max_ring": MAX_RING,
    }
    (out / "history_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results -> {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Our tiles in which AMW ever detected mining, and their neighbours.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--amw-dir", type=Path, default=compare.AMW_DIR, help=f"default: {compare.AMW_DIR}")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    history_tiles(run, args.amw_dir)
