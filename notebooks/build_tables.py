"""The prepared tables that the analysis notebooks read: <run>/analysis/.

    python notebooks/build_tables.py --run <run>              build what is new or out of date
    python notebooks/build_tables.py --run <run> --rebuild    build everything again

A period of the panel has about 145 million chips, far too many to read while a notebook
opens. This script reads each period once, in pieces of a million rows, and keeps only
what the notebooks need: counts per tile, histograms, and the full rows of the few chips
that ever came close to a detection. Each period gets its own folder, so adding a period
(2024, say) adds a folder and leaves the others as they are.

What it writes (every file is described in notebooks/README.md):

    <run>/analysis/
        _build.json                   what was built from what, and when
        static/tiles.parquet          one row per tile: centre, subregion, country, map cell
        static/chips.parquet          one row per chip, in the panel's order: distance to the
                                      nearest known mine (km), in_region, in_andes
        static/known.parquet          the known mines: chip, first year, source (AMW or ours)
        static/amw.parquet            every chip in AMW's raw files: file, confidence, rules
        static/followed.parquet       the chips followed through every period (see `hits`)
        <frequency>/period=<tag>/
            tiles.parquet             one row per tile: cloud means, cloud shares, detections
            bins.parquet              chip counts by score bin x cloud band x distance band
                                      x subregion, with sums for means and spreads
            hist.parquet              histograms of the cloud measures, per subregion
            hits.parquet              the full row of every followed chip
            sample.parquet            the full row of one chip in 1,000 (the same chips
                                      in every period), for distributions and correlations

Which detections count as "known mines" is set in notebooks/analysis.yaml: AMW's yearly
files for 2018 to 2024 for now, to be replaced year by year with our own quarters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
import panel_views as pv  # noqa: E402
from periods import Period  # noqa: E402
from analysis_tables import DIST_EDGES, EMPTY_EDGES, NCLEAR_EDGES, SCORE_EDGES  # noqa: E402

VERSION = 1                     # bump when a table's content changes: every period is rebuilt
CHUNK = 1_000_000               # rows read at a time
VOTES = [f"vote_{i}" for i in range(1, 7)]
CLOUD = ["n_obs", "n_clear", "n_clear_median", "empty_share", "thin_share", "clear_day_mean",
         "cs_mean", "cs_clear_mean", "cirrus_mean", "ndvi_mean", "bsi_mean"]
RULES = ["det_raw", "det_andes", "det_loose", "det_stringent"]
FOLLOW_FROM = 0.2               # a chip that ever scores this high is followed in every period
SAMPLE_EVERY = 1000             # one chip in this many is in the sample
CELL_DEG = 0.25                 # map cells for the overview maps
EARTH_KM = 6371.0088

# The bins (left edges; v falls in bin i when edges[i] <= v < edges[i + 1]) live in the reader,
# panel_pipeline/lib/analysis_tables.py, so the notebooks label them the same way.
HIST = {                                            # variable: (bin width, upper end of the last bin)
    "n_clear": (0.5, 100.0), "n_clear_median": (1.0, 100.0), "empty_share": (0.02, 1.0),
    "thin_share": (0.02, 1.0), "cs_mean": (0.02, 1.0), "cirrus_mean": (0.002, 0.1),
    "confidence": (0.01, 1.0),
}


# ------------------------------------------------------------------ small helpers

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def write(df: pd.DataFrame, path: Path) -> None:
    """Write a table so it appears complete or not at all (a crash leaves no half file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"_{path.name}.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def band(values: np.ndarray, edges) -> np.ndarray:
    """Bin index of each value (0 .. len(edges) - 2); -1 where the value is missing."""
    v = np.asarray(values, dtype=np.float64)
    out = np.searchsorted(np.asarray(edges, dtype=np.float64), v, side="right") - 1
    out = np.clip(out, 0, len(edges) - 2)
    out[~np.isfinite(v)] = -1
    return out.astype(np.int16)


def in_sample(chip_id: np.ndarray) -> np.ndarray:
    """One chip in SAMPLE_EVERY, picked by a fixed scramble of its ID: the same chips every time."""
    with np.errstate(over="ignore"):
        h = (chip_id.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(33)
    return (h % np.uint64(SAMPLE_EVERY)) == 0


def unit_xyz(lon, lat) -> np.ndarray:
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    """Straight-line distance between two points on the unit sphere -> distance along the Earth."""
    return 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def aligned(streams: list[tuple[Path, list[str]]], chunk: int = CHUNK):
    """Read several files with the same rows in the same order side by side, `chunk` rows at a time.

    Yields one dict {column: numpy array} per chunk. Files may cut their row groups at
    different places; this re-cuts them so every chunk holds the same rows from each.
    """
    iters = [pq.ParquetFile(p).iter_batches(batch_size=chunk, columns=cols) for p, cols in streams]
    buffers: list[list[pa.RecordBatch]] = [[] for _ in streams]
    held = [0] * len(streams)
    done = False
    while not done:
        for i, it in enumerate(iters):
            while held[i] < chunk:
                b = next(it, None)
                if b is None:
                    break
                buffers[i].append(b)
                held[i] += b.num_rows
        n = min(held)
        if n == 0:
            if any(held):
                raise ValueError("the files have different numbers of rows")
            return
        done = n < chunk
        out = {}
        for i, (path, cols) in enumerate(streams):
            t = pa.Table.from_batches(buffers[i])
            head, rest = t.slice(0, n), t.slice(n)
            buffers[i] = rest.to_batches()
            held[i] = rest.num_rows
            for c in cols:
                a = head.column(c)
                if c == "chip_id":
                    ids = a.to_numpy(zero_copy_only=False).astype(np.int64)
                    if i == 0:
                        out["chip_id"] = ids
                    elif not np.array_equal(ids, out["chip_id"]):
                        raise ValueError(f"{path} is not in the same chip order as the panel")
                elif pa.types.is_boolean(a.type):
                    out[c] = a.fill_null(False).to_numpy(zero_copy_only=False)
                else:
                    out[c] = a.cast(pa.float64()).to_numpy(zero_copy_only=False)
        yield out
        if done and any(held):
            raise ValueError("the files have different numbers of rows")
    return


# ------------------------------------------------------------------ configuration

def load_settings() -> dict:
    s = yaml.safe_load((HERE / "analysis.yaml").read_text())
    s["known_mines"]["amw_years"] = [int(y) for y in s["known_mines"].get("amw_years") or []]
    s["known_mines"]["our_years"] = [int(y) for y in s["known_mines"].get("our_years") or []]
    overlap = set(s["known_mines"]["amw_years"]) & set(s["known_mines"]["our_years"])
    if overlap:
        raise SystemExit(f"BUILD ERROR: analysis.yaml names {sorted(overlap)} under both amw_years and "
                         f"our_years; each year comes from one source")
    return s


# ------------------------------------------------------------------ static tables

def build_tiles(run: Path, out: Path) -> pd.DataFrame:
    """One row per tile: number, centre, subregion, country, map cell, chips."""
    listed = pd.read_csv(run / "plan" / "tiles.csv")
    sys.path.insert(0, str(REPO / "amw_comparison"))
    from compare import key_number          # tile key -> the tile's number (chip_id // 1000)
    listed["tile"] = [key_number(k) for k in listed["tile_key"]]
    per_tile = (listed.groupby("tile")
                .agg(tile_key=("tile_key", "first"), subregion=("subregion", "min"),
                     border=("subregion", "size"))
                .reset_index())
    per_tile["border"] = per_tile["border"] > 1

    sums = {}
    for batch in pq.ParquetFile(run / "plan" / "chips.parquet").iter_batches(
            batch_size=CHUNK, columns=["chip_id", "minx", "miny", "maxx", "maxy", "in_region", "in_andes"]):
        d = batch.to_pandas()
        d["tile"] = d["chip_id"] // 1000
        d["lon"], d["lat"] = (d["minx"] + d["maxx"]) / 2, (d["miny"] + d["maxy"]) / 2
        g = d.groupby("tile").agg(lon=("lon", "sum"), lat=("lat", "sum"), chips=("chip_id", "size"),
                                  in_region=("in_region", "sum"), in_andes=("in_andes", "sum"))
        for t, row in g.iterrows():
            if t in sums:
                sums[t] += row.to_numpy()
            else:
                sums[t] = row.to_numpy().astype(np.float64)
    s = pd.DataFrame.from_dict(sums, orient="index", columns=["lon", "lat", "chips", "in_region", "in_andes"])
    s["lon"], s["lat"] = s["lon"] / s["chips"], s["lat"] / s["chips"]
    s["in_region"] = s["in_region"] / s["chips"]
    s["in_andes"] = s["in_andes"] / s["chips"]
    tiles = per_tile.merge(s.rename_axis("tile").reset_index(), on="tile", how="inner")
    tiles["chips"] = tiles["chips"].astype(np.int32)

    from history import countries_of        # amw_comparison/history.py: nearest country outline
    tiles["country"] = countries_of(tiles["lon"].to_numpy(), tiles["lat"].to_numpy())
    tiles["cell_lon"] = (np.floor(tiles["lon"] / CELL_DEG) * CELL_DEG + CELL_DEG / 2).round(4)
    tiles["cell_lat"] = (np.floor(tiles["lat"] / CELL_DEG) * CELL_DEG + CELL_DEG / 2).round(4)
    tiles = tiles.sort_values("tile").reset_index(drop=True)
    write(tiles, out / "tiles.parquet")
    say(f"tiles: {len(tiles):,}")
    return tiles


def read_amw(folder: Path) -> pd.DataFrame:
    """Every row of AMW's raw files (confidence > 0.4), with AMW's own loose and stringent rules.

    One row per (file, location). The rules are applied per file exactly as AMW's
    postprocess does (core/postprocess.py dual_threshold_filter): loose = 0.43, and 0.75
    when the 5th nearest other chip at >= 0.43 is more than 3 km away; stringent = 0.55 / 0.8.
    """
    import geopandas as gpd
    sys.path.insert(0, str(REPO / "amw_comparison"))
    from history import key_int, raw_files, period_year
    sys.path.append(str(REPO / "core"))
    from postprocess import centroid_key, dual_threshold_filter

    files = raw_files(folder)
    parts = []
    for tag, path in files.items():
        gdf = gpd.read_file(path, columns=["confidence"])
        kx, ky = centroid_key(gdf)
        key = key_int(kx, ky)
        loose = dual_threshold_filter(gdf, t_main=0.43, t_iso=0.75, k=5, isolation_km=3.0)
        stringent = dual_threshold_filter(gdf, t_main=0.55, t_iso=0.8, k=5, isolation_km=3.0)
        yearly = tag.isdigit()
        period = tag if yearly else f"{period_year(tag)}Q{tag[1]}"
        part = pd.DataFrame({"key": key, "period": period, "frequency": "yearly" if yearly else "quarterly",
                             "year": period_year(tag), "amw_confidence": gdf["confidence"].to_numpy(np.float32),
                             "det_loose": gdf.index.isin(loose.index), "det_stringent": gdf.index.isin(stringent.index)})
        # two rows on one location (AMW dedupes them; a file written before that may not have): keep the higher
        part = part.sort_values("amw_confidence", ascending=False).drop_duplicates("key")
        parts.append(part)
        say(f"  AMW {path.name}: {len(part):,} locations, {int(part['det_loose'].sum()):,} loose")
    return pd.concat(parts, ignore_index=True)


def build_amw(run: Path, folder: Path, out: Path) -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "amw_comparison"))
    from history import chips_at
    rows = read_amw(folder)
    chips = chips_at(run, rows["key"].to_numpy())
    amw = chips.merge(rows, on="key").drop(columns="key")
    # a location in two overlapping tiles (UTM seams) maps to two chips; both are kept
    amw = amw.sort_values(["chip_id", "period"]).reset_index(drop=True)
    write(amw, out / "amw.parquet")
    say(f"AMW: {len(amw):,} rows on {amw['chip_id'].nunique():,} chips")
    return amw


def build_known(run: Path, settings: dict, amw: pd.DataFrame, out: Path) -> pd.DataFrame:
    """The known mines: chips detected (loose rule) in the chosen years, by AMW or by us.

    AMW's yearly files for the years in amw_years; our own quarterly panel for the years in
    our_years (a chip counts if any quarter of the year detected it). This is where the
    move from AMW's history to our own scoring happens: move a year from one list to the other.
    """
    k = settings["known_mines"]
    parts = []
    if k["amw_years"]:
        a = amw[(amw["frequency"] == "yearly") & amw["year"].isin(k["amw_years"]) & amw["det_loose"]]
        parts.append(a[["chip_id", "year"]].assign(source="AMW yearly"))
    panel = run / "panel"
    for year in k["our_years"]:
        quarters = [t for t in pv.tags(panel, "quarterly") if t.startswith(str(year))]
        if len(quarters) < 4:
            raise SystemExit(f"BUILD ERROR: analysis.yaml takes {year} from our quarters, but only "
                             f"{quarters or 'none'} are assembled")
        for q in quarters:
            ids = pq.read_table(pv.layer_path(panel, "quarterly", "scores", q), columns=["chip_id"]).column(0).to_numpy()
            det = pv.column(panel, "quarterly", "postprocess", q, "det_loose")
            parts.append(pd.DataFrame({"chip_id": ids[det], "year": year, "source": "our quarters"}))
    if not parts:
        known = pd.DataFrame({"chip_id": pd.Series(dtype=np.int64), "first_year": pd.Series(dtype=np.int16),
                              "years": pd.Series(dtype=np.int16), "source": pd.Series(dtype=str)})
    else:
        allk = pd.concat(parts, ignore_index=True)
        known = (allk.groupby("chip_id")
                 .agg(first_year=("year", "min"), years=("year", "nunique"),
                      source=("source", lambda s: " + ".join(sorted(set(s)))))
                 .reset_index())
    write(known, out / "known.parquet")
    say(f"known mines: {len(known):,} chips")
    return known


def build_chips(run: Path, tiles: pd.DataFrame, known: pd.DataFrame, out: Path) -> None:
    """Per chip, in the panel's order: distance (km) from its centre to the nearest known-mine chip."""
    from scipy.spatial import cKDTree
    keep = np.sort(tiles["tile"].to_numpy())
    if len(known):
        centres = pv.chip_facts(run, known["chip_id"].to_numpy())
        tree = cKDTree(unit_xyz(centres["lon"], centres["lat"]))
    else:
        tree = None
    writer, n = None, 0
    path = out / "chips.parquet"
    tmp = path.with_name("_chips.parquet.tmp")
    for batch in pq.ParquetFile(run / "plan" / "chips.parquet").iter_batches(
            batch_size=CHUNK, columns=["chip_id", "minx", "miny", "maxx", "maxy", "in_region", "in_andes"]):
        d = batch.to_pandas()
        d = d[np.isin(d["chip_id"].to_numpy() // 1000, keep)]     # as assemble_period: tiles of tiles.csv
        if tree is not None:
            chord, _ = tree.query(unit_xyz((d["minx"] + d["maxx"]) / 2, (d["miny"] + d["maxy"]) / 2), k=1)
            dist = chord_to_km(chord).astype(np.float32)
        else:
            dist = np.full(len(d), np.nan, dtype=np.float32)
        t = pa.table({"chip_id": d["chip_id"].to_numpy(), "dist_km": dist,
                      "in_region": d["in_region"].to_numpy(), "in_andes": d["in_andes"].to_numpy()})
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        n += len(d)
    writer.close()
    tmp.replace(path)
    say(f"chips: {n:,}, distance to the nearest of {len(known):,} known-mine chips")


# ------------------------------------------------------------------ one period

def summarise_period(run: Path, frequency: str, tag: str, tiles: pd.DataFrame, static: Path, dest: Path) -> list[int]:
    """tiles, bins, hist and sample of one period. Returns the chips scoring >= FOLLOW_FROM."""
    panel = run / "panel"
    scores_cols = ["chip_id", "scored", "confidence", *VOTES, *CLOUD]
    post_cols = ["chip_id", *RULES]
    tile_ids = tiles["tile"].to_numpy()
    sub_of_tile = tiles["subregion"].to_numpy()
    n_t = len(tile_ids)

    # per-tile sums
    T = {k: np.zeros(n_t) for k in ["chips", "scored", "n_clear_sum", "n_clear0", "n_clear_lt1", "empty_sum",
                                   "empty_n", "empty_half", "cs_sum", "cs_n", "thin_sum", "thin_n",
                                   "raw", "andes", "loose", "stringent", "near_raw", "vote_sd_sum", "conf_sum"]}
    T["conf_max"] = np.full(n_t, -1.0)
    # bins: subregion x score bin x n_clear band x empty band x distance band
    shape = (8, len(SCORE_EDGES) - 1 + 1, len(NCLEAR_EDGES) - 1 + 1, len(EMPTY_EDGES) - 1 + 1, len(DIST_EDGES) - 1 + 1)
    size = int(np.prod(shape))
    B = {k: np.zeros(size) for k in ["chips", "raw", "andes", "loose", "stringent",
                                     "conf_sum", "conf_sq", "vote_sd_sum", "vote_sd_sq", "n_clear_sum"]}
    H = {(s, v): np.zeros(int(np.ceil(top / w)) + 1) for s in range(8) for v, (w, top) in HIST.items()}
    follow, sample_parts = [], []

    streams = [(pv.layer_path(panel, frequency, "scores", tag), scores_cols),
               (pv.layer_path(panel, frequency, "postprocess", tag), post_cols),
               (static / "chips.parquet", ["chip_id", "dist_km", "in_region", "in_andes"])]
    rows = 0
    for c in aligned(streams):
        ids = c["chip_id"]
        n = len(ids)
        rows += n
        tile = ids // 1000
        ti = np.searchsorted(tile_ids, tile)
        if (ti >= n_t).any() or (tile_ids[np.minimum(ti, n_t - 1)] != tile).any():
            raise ValueError(f"{frequency} {tag}: chips of tiles that are not in plan/tiles.csv")
        sub = sub_of_tile[ti].astype(np.int64)
        scored = c["scored"] & np.isfinite(c["confidence"])
        conf = np.where(scored, c["confidence"], np.nan)
        votes = np.column_stack([c[v] for v in VOTES])
        vote_sd = np.where(scored, votes.std(axis=1), np.nan)
        nclear = np.where(scored, c["n_clear"], np.nan)
        empty = np.where(scored, c["empty_share"], np.nan)
        cs = np.where(scored, c["cs_mean"], np.nan)
        thin = np.where(scored, c["thin_share"], np.nan)

        # --- tiles: rows are sorted by chip_id, so a tile's chips are next to each other
        starts = np.flatnonzero(np.r_[True, ti[1:] != ti[:-1]])
        idx = ti[starts]

        def seg(x):
            return np.add.reduceat(np.nan_to_num(np.asarray(x, dtype=np.float64)), starts)

        def segn(x):
            return np.add.reduceat(np.isfinite(x).astype(np.float64), starts)

        T["chips"][idx] += np.diff(np.r_[starts, n])
        T["scored"][idx] += seg(scored)
        T["n_clear_sum"][idx] += seg(nclear)
        T["n_clear0"][idx] += seg(nclear == 0)
        T["n_clear_lt1"][idx] += seg(nclear < 1)
        T["empty_sum"][idx] += seg(empty)
        T["empty_n"][idx] += segn(empty)
        T["empty_half"][idx] += seg(empty > 0.5)
        T["cs_sum"][idx] += seg(cs)
        T["cs_n"][idx] += segn(cs)
        T["thin_sum"][idx] += seg(thin)
        T["thin_n"][idx] += segn(thin)
        for r, name in (("det_raw", "raw"), ("det_andes", "andes"), ("det_loose", "loose"), ("det_stringent", "stringent")):
            T[name][idx] += seg(c[r] & scored)
        T["near_raw"][idx] += seg((conf >= 0.3) & (conf <= 0.4))
        T["vote_sd_sum"][idx] += seg(vote_sd)
        T["conf_sum"][idx] += seg(conf)
        T["conf_max"][idx] = np.maximum(T["conf_max"][idx], np.maximum.reduceat(np.nan_to_num(conf, nan=-1), starts))

        # --- bins
        sb = np.where(scored, band(conf, SCORE_EDGES), -1) + 1                 # 0 = not scored
        nb = band(nclear, NCLEAR_EDGES) + 1
        eb = band(empty, EMPTY_EDGES) + 1
        db = band(c["dist_km"], DIST_EDGES) + 1
        code = np.ravel_multi_index((sub, sb, nb, eb, db), shape)
        B["chips"] += np.bincount(code, minlength=size)
        for r, name in (("det_raw", "raw"), ("det_andes", "andes"), ("det_loose", "loose"), ("det_stringent", "stringent")):
            B[name] += np.bincount(code, weights=(c[r] & scored).astype(float), minlength=size)
        B["conf_sum"] += np.bincount(code, weights=np.nan_to_num(conf), minlength=size)
        B["conf_sq"] += np.bincount(code, weights=np.nan_to_num(conf) ** 2, minlength=size)
        B["vote_sd_sum"] += np.bincount(code, weights=np.nan_to_num(vote_sd), minlength=size)
        B["vote_sd_sq"] += np.bincount(code, weights=np.nan_to_num(vote_sd) ** 2, minlength=size)
        B["n_clear_sum"] += np.bincount(code, weights=np.nan_to_num(nclear), minlength=size)

        # --- histograms per subregion (0 = all subregions together)
        for v, (w, top) in HIST.items():
            x = conf if v == "confidence" else np.where(scored, c[v], np.nan)
            ok = np.isfinite(x)
            k = np.minimum(np.floor(np.clip(x[ok], 0, None) / w).astype(np.int64), len(H[(0, v)]) - 1)
            H[(0, v)] += np.bincount(k, minlength=len(H[(0, v)]))
            for s in np.unique(sub[ok]):
                m = sub[ok] == s
                H[(int(s), v)] += np.bincount(k[m], minlength=len(H[(0, v)]))

        follow.append(ids[conf >= FOLLOW_FROM])
        pick = in_sample(ids)
        if pick.any():
            sample_parts.append(pd.DataFrame({"chip_id": ids[pick], "subregion": sub[pick].astype(np.int8),
                                              "scored": scored[pick], "confidence": conf[pick],
                                              "vote_sd": vote_sd[pick],
                                              **{v: c[v][pick].astype(np.float32) for v in VOTES + CLOUD},
                                              **{r: np.where(scored[pick], c[r][pick], False) for r in RULES},
                                              "dist_km": c["dist_km"][pick].astype(np.float32),
                                              "in_region": c["in_region"][pick], "in_andes": c["in_andes"][pick]}))

    # --- write
    t = pd.DataFrame({"tile": tile_ids, "chips": T["chips"].astype(np.int32), "scored": T["scored"].astype(np.int32)})
    sc = np.maximum(T["scored"], 1)
    t["n_clear_mean"] = np.where(T["scored"] > 0, T["n_clear_sum"] / sc, np.nan).astype(np.float32)
    t["share_n_clear_0"] = np.where(T["scored"] > 0, T["n_clear0"] / sc, np.nan).astype(np.float32)
    t["share_n_clear_under_1"] = np.where(T["scored"] > 0, T["n_clear_lt1"] / sc, np.nan).astype(np.float32)
    t["empty_share_mean"] = np.where(T["empty_n"] > 0, T["empty_sum"] / np.maximum(T["empty_n"], 1), np.nan).astype(np.float32)
    t["share_empty_over_half"] = np.where(T["empty_n"] > 0, T["empty_half"] / np.maximum(T["empty_n"], 1), np.nan).astype(np.float32)
    t["cs_mean_mean"] = np.where(T["cs_n"] > 0, T["cs_sum"] / np.maximum(T["cs_n"], 1), np.nan).astype(np.float32)
    t["thin_share_mean"] = np.where(T["thin_n"] > 0, T["thin_sum"] / np.maximum(T["thin_n"], 1), np.nan).astype(np.float32)
    for name in ("raw", "andes", "loose", "stringent", "near_raw"):
        t[name] = T[name].astype(np.int32)
    t["confidence_mean"] = np.where(T["scored"] > 0, T["conf_sum"] / sc, np.nan).astype(np.float32)
    t["confidence_max"] = np.where(T["conf_max"] >= 0, T["conf_max"], np.nan).astype(np.float32)
    t["vote_sd_mean"] = np.where(T["scored"] > 0, T["vote_sd_sum"] / sc, np.nan).astype(np.float32)
    write(t[t["chips"] > 0], dest / "tiles.parquet")

    nz = np.flatnonzero(B["chips"])
    s_, sb_, nb_, eb_, db_ = np.unravel_index(nz, shape)
    bins = pd.DataFrame({"subregion": s_.astype(np.int8), "score_bin": (sb_ - 1).astype(np.int8),
                         "n_clear_band": (nb_ - 1).astype(np.int8), "empty_band": (eb_ - 1).astype(np.int8),
                         "dist_band": (db_ - 1).astype(np.int8)})
    for k, v in B.items():
        bins[k] = v[nz] if k.endswith(("sum", "sq")) else v[nz].astype(np.int64)
    write(bins, dest / "bins.parquet")

    hist = []
    for (s, v), counts in H.items():
        w, top = HIST[v]
        nzh = np.flatnonzero(counts)
        hist.append(pd.DataFrame({"subregion": np.int8(s), "variable": v, "left": nzh * w,
                                  "right": np.where(nzh == len(counts) - 1, np.inf, (nzh + 1) * w),
                                  "chips": counts[nzh].astype(np.int64)}))
    write(pd.concat(hist, ignore_index=True), dest / "hist.parquet")

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    write(sample, dest / "sample.parquet")
    say(f"  {frequency} {tag}: {rows:,} chips summarised")
    return np.unique(np.concatenate(follow)).tolist() if follow else []


def extract_hits(run: Path, frequency: str, tag: str, followed: pd.DataFrame, dest: Path) -> None:
    """The full row of every followed chip in this period (both layers, plus the vote spread)."""
    panel = run / "panel"
    rows = pv.rows_of_chips(panel, frequency, tag, followed["chip_id"].to_numpy(),
                            columns={"scores": ["scored", "confidence", *VOTES, *CLOUD], "postprocess": RULES})
    rows = rows.drop(columns="period")
    rows["scored"] = rows["scored"].fillna(False).astype(bool) & rows["confidence"].notna()
    rows["vote_sd"] = rows[VOTES].astype(np.float64).std(axis=1, ddof=0).where(rows["scored"]).astype(np.float32)
    for r in RULES:
        rows[r] = rows[r].astype("boolean").fillna(False).astype(bool) & rows["scored"]
    write(rows.sort_values("chip_id").reset_index(drop=True), dest / "hits.parquet")


# ------------------------------------------------------------------ the run

def build(run: Path, amw_dir: Path, rebuild: bool = False) -> Path:
    t0 = time.time()
    settings = load_settings()
    panel = run / "panel"
    out = run / "analysis"
    static = out / "static"
    out.mkdir(exist_ok=True)
    log_path = out / "_build.json"
    log = json.loads(log_path.read_text()) if log_path.exists() and not rebuild else {}
    if log.get("version") != VERSION:
        log = {}

    periods = {f: pv.tags(panel, f) for f in ("quarterly", "monthly")}
    if not any(periods.values()):
        raise SystemExit(f"BUILD ERROR: nothing assembled in {panel}")
    say(f"run {run.name}: {len(periods['quarterly'])} quarters, {len(periods['monthly'])} months assembled")

    # --- static tables: rebuilt when the plan, AMW's files or the known-mine settings change
    sys.path.insert(0, str(REPO / "amw_comparison"))
    from history import raw_files
    from compare import sha256
    files = raw_files(amw_dir)
    amw_print = {t: sha256(p)[:12] for t, p in files.items()}
    plan_print = [(run / "plan" / f).stat().st_size for f in ("tiles.csv", "chips.parquet")]
    our_records = {y: [pv.record(panel, "quarterly", "postprocess", q)["assembled_utc"]
                       for q in periods["quarterly"] if q.startswith(str(y))]
                   for y in settings["known_mines"]["our_years"]}
    static_print = fingerprint(VERSION, plan_print, amw_print, settings["known_mines"], our_records)
    if log.get("static") != static_print or not (static / "chips.parquet").exists():
        say("static tables (tiles, AMW, known mines, distances)...")
        tiles = build_tiles(run, static)
        amw = build_amw(run, amw_dir, static)
        known = build_known(run, settings, amw, static)
        build_chips(run, tiles, known, static)
        log = {"version": VERSION, "static": static_print, "periods": {}}
        log["amw_files"] = {t: p.name for t, p in files.items()}
        log["amw_fingerprints"] = amw_print
        log["known_mines"] = settings["known_mines"]
        log["static_built_utc"] = now()
        log_path.write_text(json.dumps(log, indent=2))
    tiles = pd.read_parquet(static / "tiles.parquet")
    amw = pd.read_parquet(static / "amw.parquet")
    known = pd.read_parquet(static / "known.parquet")
    log.setdefault("periods", {})

    # --- per period summaries: rebuilt when the period's panel layers change
    for f, tags in periods.items():
        for tag in tags:
            dest = out / f / f"period={tag}"
            inputs = [pv.record(panel, f, layer, tag).get("assembled_utc") for layer in pv.LAYERS]
            key = f"{f}/{tag}"
            entry = log["periods"].get(key, {})
            if entry.get("inputs") == inputs and (dest / "sample.parquet").exists() and "follow" in entry:
                continue
            say(f"summarising {f} {tag}...")
            follow = summarise_period(run, f, tag, tiles, static, dest)
            log["periods"][key] = {"inputs": inputs, "summarised_utc": now(), "follow": len(follow)}
            write(pd.DataFrame({"chip_id": np.asarray(follow, dtype=np.int64)}), dest / "_follow.parquet")
            log_path.write_text(json.dumps(log, indent=2))

    # --- the followed chips: ever >= FOLLOW_FROM in a period, in any AMW file, or a known mine
    ids = [pd.read_parquet(out / f / f"period={t}" / "_follow.parquet")["chip_id"].to_numpy()
           for f, tags in periods.items() for t in tags]
    ids += [amw["chip_id"].to_numpy(), known["chip_id"].to_numpy()]
    chip_ids = np.unique(np.concatenate(ids)).astype(np.int64)
    followed_print = fingerprint(chip_ids.tolist()[:50], len(chip_ids), int(chip_ids.sum() % 1_000_000_007))
    if log.get("followed") != followed_print or not (static / "followed.parquet").exists():
        facts = pv.chip_facts(run, chip_ids)[["chip_id", "lon", "lat", "in_region", "in_andes"]]
        facts["tile"] = facts["chip_id"] // 1000
        facts = facts.merge(tiles[["tile", "subregion", "country"]], on="tile", how="left")
        dist = pd.read_parquet(static / "chips.parquet", columns=["chip_id", "dist_km"],
                               filters=[("chip_id", "in", chip_ids.tolist())]) if len(chip_ids) else \
            pd.DataFrame(columns=["chip_id", "dist_km"])
        facts = facts.merge(dist, on="chip_id", how="left")
        facts = facts.merge(known.rename(columns={"first_year": "known_since", "years": "known_years",
                                                  "source": "known_source"}), on="chip_id", how="left")
        facts["known"] = facts["known_since"].notna()
        write(facts.sort_values("chip_id").reset_index(drop=True), static / "followed.parquet")
        log["followed"] = followed_print
        log["followed_chips"] = int(len(chip_ids))
        log_path.write_text(json.dumps(log, indent=2))
        say(f"followed chips: {len(chip_ids):,}")
    followed = pd.read_parquet(static / "followed.parquet")

    for f, tags in periods.items():
        for tag in tags:
            dest = out / f / f"period={tag}"
            entry = log["periods"][f"{f}/{tag}"]
            if entry.get("hits") == log["followed"] and (dest / "hits.parquet").exists():
                continue
            extract_hits(run, f, tag, followed, dest)
            entry["hits"] = log["followed"]
            log_path.write_text(json.dumps(log, indent=2))
            say(f"  {f} {tag}: rows of {len(followed):,} followed chips")

    log["built_utc"] = now()
    log["periods_built"] = {f: tags for f, tags in periods.items()}
    log_path.write_text(json.dumps(log, indent=2))
    say(f"done -> {out} ({(time.time() - t0) / 60:.1f} min)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The prepared tables the analysis notebooks read.")
    ap.add_argument("--run", required=True, type=Path, help="the run folder")
    ap.add_argument("--amw-dir", type=Path, default=REPO / "amw_comparison" / "data" / "amw",
                    help="AMW's raw files (amazon_basin_<period>_t0.4.geojson)")
    ap.add_argument("--rebuild", action="store_true", help="build everything again")
    a = ap.parse_args()
    build(a.run.expanduser().resolve(), a.amw_dir.expanduser().resolve(), a.rebuild)
