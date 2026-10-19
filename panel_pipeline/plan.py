"""Plan a run: make its folder, record its settings, list every tile and chip.

Run once per run, before any scoring:
    python panel_pipeline/plan.py
    python panel_pipeline/plan.py --config path/to/other.yaml

Writes <output_root>/<run.name>/
    run/config.yaml       exact copy of the settings; every later script reads THIS copy
    run/provenance.json   code version, file fingerprints, package versions, counts
    plan/tiles.csv        one row per (subregion, tile): AMW's six tile lists, stacked
    plan/chips.parquet    one row per chip: chip_id, square, AMW key, in_region, in_andes
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import yaml

from config import DEFAULT_CONFIG, Config, describe, load_config
from grid import CHIP_PX, chip_table, read_shape, tiles_for
from provenance import environment, fingerprints


# ------------------------------------------------------------ tiles

def plan_tiles(cfg: Config):
    """AMW's tile list per subregion, plus what the chip flags need."""
    clip = read_shape(cfg.region.clip_to) if cfg.region.clip_to else None
    shapes, rows, tiles = [], [], {}
    for n, path in cfg.region.subregion_files.items():
        shape = read_shape(path)
        shapes.append(shape)
        if clip is not None:
            # DEVIATION (test runs only): AMW tests by pointing inference at a
            # different region file. We cut each subregion down to the test
            # area instead, so a test goes through the same subregion routing
            # as the full run. clip_to: null gives AMW's behaviour exactly.
            shape = shape.intersection(clip)
            # Where the test area only touches this subregion's border, the cut
            # leaves lines or points. They have no area to tile, and AMW's tiler
            # (DLTile) refuses them, so keep only the polygons.
            shape = shapely.union_all([p for p in shapely.get_parts(shape)
                                       if p.geom_type in ("Polygon", "MultiPolygon")])
        for tile in tiles_for(shape, cfg.imagery):
            # Same as AMW: a tile on a subregion border is listed under both
            # subregions, is scored twice, and the copies are merged later
            # (dedupe, postprocess.py:146).
            rows.append({"subregion": n, "tile_key": tile.key})
            tiles[tile.key] = tile
    tile_list = pd.DataFrame(rows, columns=["subregion", "tile_key"])

    # AMW's Andes pass runs inference over the Andes boundary file by itself
    # (pipeline.py:163-168), which tiles it like any other region.
    andes = read_shape(cfg.region.andes_boundary)
    andes_keys = {t.key for t in tiles_for(andes, cfg.imagery)}
    if clip is None and not andes_keys <= set(tiles):
        # The Andes area lies inside Amazon_ACA, so every Andes tile must
        # already be in a subregion list; if not, a boundary file is wrong.
        raise SystemExit(f"PLAN ERROR: {len(andes_keys - set(tiles))} Andes tiles "
                         f"are in no subregion list; check the boundary files")
    return tile_list, tiles, shapely.union_all(shapes), andes, andes_keys

# ------------------------------------------------------------ chips

def write_chips(path: Path, tiles: dict, stride: int, region, andes,
                andes_keys: set, batch_tiles: int) -> dict:
    """Write one row per chip, a batch of tiles at a time, and count as we go."""
    shapely.prepare(region)          # pre-index the big polygons: many fast tests
    shapely.prepare(andes)
    ordered = sorted(tiles.values(), key=lambda t: (t.zone, t.ti, t.tj))
    n_batches = math.ceil(len(ordered) / batch_tiles)
    writer, last_id = None, -1
    counts = {"chips": 0, "in_region": 0, "in_andes": 0}
    for b in range(n_batches):
        batch = ordered[b * batch_tiles:(b + 1) * batch_tiles]
        chips = pd.concat([chip_table(t, stride) for t in batch], ignore_index=True)
        squares = chips.geometry.values

        # ADDITION (AMW has no such flag): AMW keeps every chip of every tile
        # that touches the region, even chips just outside it. We keep them too,
        # and flag which ones touch the region, so the analysis can restrict the
        # sample. "Touch" is AMW's own clipping rule (geo_filter.py:30).
        chips["in_region"] = shapely.intersects(region, squares)
        # AMW's Andes chips: chips of AMW's Andes tiles that touch the Andes
        # boundary (pipeline.py:395 runs geo_filter.py with that rule).
        chips["in_andes"] = (chips["tile_key"].isin(andes_keys)
                             & shapely.intersects(andes, squares))

        # Check: IDs strictly increasing, within the batch and from the last
        # batch, which proves no two chips share an ID without holding them all.
        ids = chips["chip_id"].to_numpy()
        if ids[0] <= last_id or (ids[1:] <= ids[:-1]).any():
            raise SystemExit("PLAN ERROR: chip_ids are not unique and increasing")
        last_id = ids[-1]

        # Stored at full precision. AMW rounds file coordinates to 6 decimals
        # (postprocess.py:62-81) because it joins periods on coordinates; we
        # join on chip_id, so we keep the squares exactly as computed.
        table = pa.Table.from_pandas(chips.drop(columns="geometry"),
                                     preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        counts["chips"] += len(chips)
        counts["in_region"] += int(chips["in_region"].sum())
        counts["in_andes"] += int(chips["in_andes"].sum())
        print(f"  chips: batch {b + 1}/{n_batches}, {counts['chips']:,} so far", flush=True)
    if writer is not None:
        writer.close()
    return counts


# ------------------------------------------------------------ the run folder

def plan(cfg: Config) -> tuple[Path, dict]:
    """Build the run folder under a temporary name; rename it when complete."""
    run_dir = cfg.output_root / cfg.run_name
    if run_dir.exists():
        raise SystemExit(f"PLAN ERROR: {run_dir} already exists. A run is planned "
                         f"once: choose a new run.name, or delete that folder.")
    print("tiles ...", flush=True)
    tile_list, tiles, region, andes, andes_keys = plan_tiles(cfg)
    if not tiles:
        raise SystemExit("PLAN ERROR: no tiles; does clip_to overlap the subregions?")

    work = run_dir.with_name(run_dir.name + ".planning")
    if work.exists():
        shutil.rmtree(work)              # left by a crashed attempt; never valid
    (work / "run").mkdir(parents=True)
    (work / "plan").mkdir()
    shutil.copyfile(cfg.source, work / "run" / "config.yaml")
    tile_list.to_csv(work / "plan" / "tiles.csv", index=False)

    stride = CHIP_PX // cfg.inference.stride_ratio      # AMW: inference_engine.py:756
    counts = write_chips(work / "plan" / "chips.parquet", tiles, stride, region,
                         andes, andes_keys, cfg.inference.batch_tiles)
    per_sub = tile_list.groupby("subregion").size()
    counts |= {
        "tiles_per_subregion": {int(n): int(c) for n, c in per_sub.items()},
        "tiles_unique": len(tiles),
        "tiles_on_subregion_borders": int((tile_list["tile_key"].value_counts() > 1).sum()),
        "andes_tiles_planned": len(andes_keys & set(tiles)),
    }

    # What made this run (provenance.py), then what it contains.
    record = ({"planned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
              | environment(cfg) | {"file_sha256": fingerprints(cfg), "counts": counts})
    (work / "run" / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    work.rename(run_dir)                 # the run folder appears only when complete
    return run_dir, record

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan a run: folder, settings copy, tiles, chips.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to a config.yaml (default: the one next to this file)")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")
    print(describe(cfg) + "\n", flush=True)

    run_dir, record = plan(cfg)
    c = record["counts"]
    per_sub = ", ".join(f"{n}: {k:,}" for n, k in c["tiles_per_subregion"].items())
    print(f"\ntiles        per subregion {per_sub}; {c['tiles_unique']:,} unique, "
          f"{c['tiles_on_subregion_borders']:,} on subregion borders")
    print(f"chips        {c['chips']:,}, of which {c['in_region']:,} touch the region "
          f"and {c['in_andes']:,} are Andes chips")
    if record["git_uncommitted_files"]:
        print(f"WARNING      {len(record['git_uncommitted_files'])} file(s) differ from the "
              f"last commit (see provenance.json); commit before a real run")
    print(f"\nplanned  ->  {run_dir}")