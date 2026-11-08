"""Is the quarter's last day the whole difference between our scores and AMW's? A full check.

    python amw_comparison/last_day_check.py --run <run> --year 2026
    python amw_comparison/last_day_check.py --run <run> --quarters 2026Q1 --tiles 4 --subregions 4 6

AMW's quarter leaves out its last day (last_day.py), ours keeps it. This script tests that
explanation three ways, for each quarter, and writes what it finds to
<run>/amw_comparison/last_day_check/<quarter>/:

1. by_subregion.csv  Per subregion and outcome (both with the same score / both with
   different scores / only ours / only AMW): chips whose centre lies under a scene taken
   on the last day, chips whose square only touches one, and chips elsewhere. Compare
   each share with the "both, same score" row of the same subregion: that is how much of
   the subregion's detections the last day's scenes cover anyway.
2. outside.csv       The chips that differ although their centre is not under such a
   scene: how big the gap is, whether the chip's square touches a scene, how far its
   centre is from the nearest one.
3. rescore.csv       Re-scoring, the decisive test. A few tiles per subregion that hold
   disagreeing chips are downloaded and scored again three times:
     a. AMW's own downloader (core/gee.py, 64-bit) with AMW's end date
        -> must give AMW's published scores, and the same chips above 0.4;
     b. our downloader (detect.py, 32-bit) with our end date
        -> must give the scores in our detect output;
     c. AMW's own downloader (64-bit) with our end date
        -> must also give our scores: the number format is not the difference.
   The scoring is AMW's, line for line (inference_engine.py:945-949, 766, 786).
   rescore_chips.parquet holds every chip of those tiles with all five scores.
summary.txt says, in a few lines, whether each test passed.

Needs compare.py's results for the quarter (chips_<quarter>.parquet), Earth Engine (the
last day's scene outlines, and 3 tile downloads per tile re-scored) and TensorFlow.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import shapely

import compare
from compare import REPO

TOLERANCE = 1e-5          # scores this close count as the same (one run vs another of the
                          # same model on the same pixels differs by about 1e-7)
OUTCOMES = ["both, same score", "both, scores differ", "only ours", "only AMW"]
KM_PER_DEGREE = 111.32    # at the equator; distances below are "about", which is enough here


# ------------------------------------------------------------ 1 and 2: where the gaps are

def label(chips: pd.DataFrame) -> pd.Series:
    """Each chip's outcome; chips both detected with different scores get their own label."""
    out = compare.outcome(chips).astype(object)
    both = out == "both"
    differ = (chips["confidence"] - chips["amw_confidence"]).abs() > 1e-6
    out[both & differ] = "both, scores differ"
    out[both & ~differ] = "both, same score"
    return out                     # and "neither": detected only in one of our months


def squares(run: Path, chip_ids: np.ndarray) -> pd.DataFrame:
    """The chips' squares (minx, miny, maxx, maxy) from the plan."""
    t = ds.dataset(run / "plan" / "chips.parquet").to_table(
        columns=["chip_id", "minx", "miny", "maxx", "maxy"],
        filter=pc.field("chip_id").isin(chip_ids))
    return t.to_pandas()


def place(chips: pd.DataFrame, outline, run: Path) -> pd.DataFrame:
    """Add to each chip: centre under a last-day scene, square touching one, distance (km)."""
    sq = squares(run, chips["chip_id"].to_numpy())
    t = chips.merge(sq, on="chip_id", how="left", validate="one_to_one")
    t["outcome"] = label(t)
    t["centre_under"] = shapely.contains_xy(outline, t["amw_key_x"], t["amw_key_y"])
    boxes = shapely.box(t["minx"], t["miny"], t["maxx"], t["maxy"])
    t["square_touches"] = shapely.intersects(boxes, outline)
    centres = shapely.points(t["amw_key_x"], t["amw_key_y"])
    t["km_to_last_day"] = (0.0 if outline.is_empty else
                           shapely.distance(centres, outline) * KM_PER_DEGREE)
    t["gap"] = (t["confidence"] - t["amw_confidence"]).abs()
    return t


def by_subregion(t: pd.DataFrame) -> pd.DataFrame:
    """Per subregion and outcome: chips under the last day's scenes, touching, elsewhere."""
    t = t.assign(where=np.select([t["centre_under"], t["square_touches"]],
                                 ["centre under", "square touches"], "elsewhere"))
    table = (t.groupby(["subregion", "outcome"])["where"].value_counts().unstack(fill_value=0)
              .reindex(columns=["centre under", "square touches", "elsewhere"], fill_value=0))
    table = table.reindex(pd.MultiIndex.from_product([sorted(t["subregion"].unique()), OUTCOMES],
                                                     names=["subregion", "outcome"]), fill_value=0)
    total = table.sum(axis=1)
    table["chips"] = total
    table["share centre under"] = (table["centre under"] / total.where(total > 0)).round(4)
    table["share under or touching"] = ((table["centre under"] + table["square touches"])
                                        / total.where(total > 0)).round(4)
    return table.reset_index()


def outside(t: pd.DataFrame) -> pd.DataFrame:
    """The chips that disagree although their centre is not under a last-day scene."""
    rows = t[t["outcome"].isin(OUTCOMES[1:]) & ~t["centre_under"]]
    return rows[["chip_id", "subregion", "outcome", "confidence", "amw_confidence", "gap",
                 "square_touches", "km_to_last_day", "n_clear"]].sort_values("gap", ascending=False)


def describe_outside(rows: pd.DataFrame) -> list[str]:
    """A few lines about those chips, for the summary."""
    if rows.empty:
        return ["  none: every disagreeing chip lies under a last-day scene"]
    differ = rows[rows["outcome"] == "both, scores differ"]
    lines = [f"  {len(rows):,} chips ({', '.join(f'{n:,} {o}' for o, n in rows['outcome'].value_counts().items())})",
             f"  square touching a last-day scene: {int(rows['square_touches'].sum()):,}; "
             f"centre within 1 km of one: {int((rows['km_to_last_day'] <= 1).sum()):,}"]
    if len(differ):
        g = differ["gap"]
        lines.append(f"  score gaps of the 'scores differ' ones: median {g.median():.2g}, largest "
                     f"{g.max():.2g}; under 0.001: {int((g < 1e-3).sum()):,}, 0.001-0.01: "
                     f"{int(((g >= 1e-3) & (g < 1e-2)).sum()):,}, 0.01 or more: {int((g >= 1e-2).sum()):,}")
    return lines


# ------------------------------------------------------------ 3: scoring tiles again

OUTSIDE = "disagreement outside the last day"
CONTROL = "control: agreement elsewhere"


def pick_tiles(t: pd.DataFrame, per_subregion: int, subregions: list[int] | None,
               seed: int = 0, outside_per_subregion: int = 0) -> pd.DataFrame:
    """Per subregion, up to N random tiles that hold chips under a last-day scene which
    disagree, plus one tile of chips that agree and lie elsewhere (a control).

    With outside_per_subregion > 0, also the M tiles with the most chips that disagree by
    0.01 or more although no last-day scene touches them: the gaps the last day cannot
    explain. Re-scoring those says whether AMW's own code, run today with AMW's dates,
    still gives AMW's file there (then the date window is the cause after all) or gives our
    scores (then the imagery in Earth Engine changed after AMW scored the quarter)."""
    rng = np.random.default_rng(seed)
    picks = []
    # Tiles where two of our chips share a location (UTM zone seams) are left out: there
    # AMW's row belongs to only one of the two, and the test would count the other as missing.
    seams = t.loc[t["border_copies"] > 1, "tile"].unique() if "border_copies" in t else []
    t = t[~t["tile"].isin(seams)]
    for n in subregions or sorted(t["subregion"].unique()):
        mine = t[t["subregion"] == n]
        disagree = mine["outcome"].isin(OUTCOMES[1:])
        bad = mine.loc[disagree & mine["centre_under"], "tile"].unique()
        good = np.setdiff1d(mine.loc[(mine["outcome"] == OUTCOMES[0]) & ~mine["square_touches"], "tile"].unique(),
                            mine.loc[disagree, "tile"].unique())
        for tile in rng.permutation(bad)[:per_subregion]:
            picks.append({"subregion": n, "tile": int(tile), "why": "disagreement under the last day"})
        for tile in rng.permutation(good)[:1]:
            picks.append({"subregion": n, "tile": int(tile), "why": CONTROL})
        if outside_per_subregion > 0:
            far = mine[disagree & ~mine["square_touches"] & (mine["gap"].fillna(1.0) >= 0.01)]
            for tile in far["tile"].value_counts().index[:outside_per_subregion]:
                picks.append({"subregion": n, "tile": int(tile), "why": OUTSIDE})
    return pd.DataFrame(picks, columns=["subregion", "tile", "why"])


def our_scores(run: Path, quarter: str, subregion: int, tile: int) -> pd.DataFrame:
    """Every chip of one tile, as our detect output holds it."""
    folder = compare.task_folder(run, quarter, subregion)
    t = ds.dataset(sorted(map(str, folder.glob("part_*.parquet")))).to_table(
        columns=["chip_id", "confidence"],
        filter=pc.divide(pc.field("chip_id"), 1000) == tile)
    return t.to_pandas().rename(columns={"confidence": "ours_stored"})


class Rescorer:
    """Downloads one tile for a date window and scores it as AMW does."""

    def __init__(self, run: Path):
        sys.path.insert(0, str(REPO / "panel_pipeline"))
        import detect                  # our downloader; loads TensorFlow and AMW's gee.py
        import gee
        import tensorflow as tf
        from config import load_config
        self.detect, self.gee = detect, gee
        self.cfg = load_config(run / "run" / "config.yaml")
        gee.EE_PROJECT = self.cfg.ee_project          # billed to the run's own project
        detect.log_in_to_earth_engine()
        self.model = tf.keras.models.load_model(str(self.cfg.inference.model_path), compile=False)
        self.stride = detect.CHIP_PX // self.cfg.inference.stride_ratio
        im = self.cfg.imagery
        self.amw_config = gee.DataConfig(tilesize=im.tilesize, pad=im.pad, collection=im.collection,
                                         clear_threshold=im.clear_threshold, max_workers=1)
        self.extractors = {}

    def extractor(self, kind: str, start: str, end: str):
        key = (kind, start, end)
        if key not in self.extractors:
            if kind == "amw":           # AMW's class, untouched: 64-bit, the 13 bands only
                self.extractors[key] = self.gee.GEE_Data_Extractor(start, end, self.amw_config)
            else:                       # ours, exactly as detect.py builds it
                period = SimpleNamespace(start=date.fromisoformat(start), end=date.fromisoformat(end))
                self.extractors[key] = self.detect.extractor_for(self.cfg, period, None)
        return self.extractors[key]

    def score(self, tile_key: str, kind: str, start: str, end: str) -> np.ndarray:
        """One tile's confidences, in chip order (chip_index 0, 1, ...)."""
        from descarteslabs.geo import DLTile
        tile = DLTile.from_key(tile_key)
        pixels = self.extractor(kind, start, end).get_tile_data(tile)
        spectral = pixels[..., :len(self.detect.SPECTRAL)]
        # AMW's scoring, line for line (inference_engine.py:945-949, 766, 786)
        normed = np.clip(spectral / 10000.0, 0, 1).astype(np.float32)
        chips, _ = self.detect.cut_chips(normed, tile.geometry.bounds, self.detect.CHIP_PX,
                                         self.stride, crs="epsg:4326")
        votes = self.model.predict(chips, verbose=0)
        return np.asarray(np.mean(votes, axis=1), dtype=np.float64)


def rescore(run: Path, quarter: str, picks: pd.DataFrame, t: pd.DataFrame, rescorer) -> tuple:
    """Score the picked tiles again three ways; per tile and per chip results."""
    import last_day
    start, amw_end, our_end = (d.isoformat() for d in last_day.windows(quarter))
    tiles = pd.read_csv(run / "plan" / "tiles.csv")
    key_of = dict(zip(tiles["tile_key"].map(compare.key_number), tiles["tile_key"]))
    amw = t.set_index("chip_id")["amw_confidence"]
    per_tile, per_chip = [], []
    for p in picks.itertuples():
        began = time.time()
        key = key_of[p.tile]
        chips = our_scores(run, quarter, p.subregion, p.tile).sort_values("chip_id", ignore_index=True)
        a = rescorer.score(key, "amw", start, amw_end)         # a. AMW's code, AMW's dates
        b = rescorer.score(key, "ours", start, our_end)        # b. our code, our dates
        c = rescorer.score(key, "amw", start, our_end)         # c. AMW's code, our dates
        if not (len(a) == len(b) == len(c) == len(chips)):
            raise RuntimeError(f"tile {key}: {len(chips)} chips stored, {len(a)} re-scored")
        chips = chips.assign(subregion=p.subregion, tile_key=key,
                             amw_published=chips["chip_id"].map(amw),   # NaN: not in AMW's file
                             amw_code_amw_dates=a, our_code_our_dates=b, amw_code_our_dates=c)
        # AMW's file holds only chips above 0.4, so "the same chips above 0.4" is the test
        # for the rest; where AMW has a row, its score must match.
        listed = chips["amw_published"].notna()
        above = chips["amw_code_amw_dates"] > compare.RAW_THRESHOLD
        gap_a = (chips["amw_code_amw_dates"] - chips["amw_published"]).abs()[listed]
        row = {"subregion": p.subregion, "tile_key": key, "why": p.why, "chips": len(chips),
               "AMW rows": int(listed.sum()),
               "a: largest gap to AMW's file": float(gap_a.max()) if len(gap_a) else 0.0,
               "a: chips above 0.4 but not in AMW's file": int((above & ~listed).sum()),
               "a: in AMW's file but not above 0.4": int((listed & ~above).sum()),
               "b: largest gap to our stored scores": float((chips["our_code_our_dates"] - chips["ours_stored"]).abs().max()),
               "c: largest gap to our stored scores": float((chips["amw_code_our_dates"] - chips["ours_stored"]).abs().max()),
               "chips whose score the last day changes": int(((chips["ours_stored"] - chips["amw_code_amw_dates"]).abs() > 1e-6).sum()),
               "minutes": round((time.time() - began) / 60, 1)}
        row["a ok"] = (row["a: largest gap to AMW's file"] <= TOLERANCE
                       and row["a: chips above 0.4 but not in AMW's file"] == 0
                       and row["a: in AMW's file but not above 0.4"] == 0)
        row["b ok"] = row["b: largest gap to our stored scores"] <= TOLERANCE
        row["c ok"] = row["c: largest gap to our stored scores"] <= TOLERANCE
        print(f"  {quarter} subregion {p.subregion} tile {key} ({p.why}): a {'ok' if row['a ok'] else 'DIFFERS'}, "
              f"b {'ok' if row['b ok'] else 'DIFFERS'}, c {'ok' if row['c ok'] else 'DIFFERS'}; "
              f"{row['chips whose score the last day changes']} chips changed by the last day "
              f"({row['minutes']} min)", flush=True)
        per_tile.append(row)
        per_chip.append(chips)
    return (pd.DataFrame(per_tile),
            pd.concat(per_chip, ignore_index=True) if per_chip else pd.DataFrame())


# ------------------------------------------------------------ one quarter

def check_quarter(run: Path, quarter: str, tiles_per_subregion: int,
                  subregions: list[int] | None, scenes_fn=None, rescorer=None,
                  outside_per_subregion: int = 0) -> list[str]:
    """All three tests for one quarter. Writes the folder; returns the summary lines."""
    import last_day
    results = run / "amw_comparison"
    chips = pd.read_parquet(results / f"chips_{quarter}.parquet")
    out = results / "last_day_check" / quarter
    out.mkdir(parents=True, exist_ok=True)

    scenes, outline = (scenes_fn or last_day.last_day_scenes)(chips, quarter)
    last = last_day.windows(quarter)[1]
    summary = [f"{quarter}: {len(scenes)} Sentinel-2 scenes taken on {last} (AMW leaves that day out)"]
    t = place(chips, outline, run)

    table = by_subregion(t)
    table.to_csv(out / "by_subregion.csv", index=False)
    summary.append("1. Share of chips whose centre lies under a last-day scene, per subregion. "
                   "Baseline: 'both, same score', how much of the detections the scenes cover anyway")
    wide = table.pivot(index="subregion", columns="outcome", values="share centre under")[OUTCOMES]
    counts = table.pivot(index="subregion", columns="outcome", values="chips")[OUTCOMES]
    for n in wide.index:
        summary.append(f"   subregion {n}: " + "; ".join(
            f"{o} {wide.loc[n, o]:.0%} of {counts.loc[n, o]:,}" if counts.loc[n, o] else f"{o} -"
            for o in OUTCOMES))

    rows = outside(t)
    rows.to_csv(out / "outside.csv", index=False)
    summary.append("2. Chips that disagree although their centre is not under a last-day scene")
    summary += describe_outside(rows)

    if tiles_per_subregion > 0 or outside_per_subregion > 0:
        picks = pick_tiles(t, tiles_per_subregion, subregions,
                           outside_per_subregion=outside_per_subregion)
        per_tile, per_chip = rescore(run, quarter, picks, t, rescorer or Rescorer(run))
        per_tile.to_csv(out / "rescore.csv", index=False)
        if len(per_chip):
            per_chip.to_parquet(out / "rescore_chips.parquet", index=False)
        summary.append(f"3. Re-scoring {len(per_tile)} tiles "
                       f"({int((per_tile['why'] != CONTROL).sum())} with "
                       f"disagreements, the rest controls), tolerance {TOLERANCE:g}")
        for test, what in (("a", "AMW's code + AMW's end date = AMW's published file"),
                           ("b", "our code + our end date = our scores"),
                           ("c", "AMW's 64-bit code + our end date = our scores")):
            ok = int(per_tile[f"{test} ok"].sum()) if len(per_tile) else 0
            summary.append(f"   {test}. {what}: {ok} of {len(per_tile)} tiles"
                           + ("" if ok == len(per_tile) else "  <- NOT ALL"))
        if len(per_tile):
            worst = [per_tile[c].max() for c in ("a: largest gap to AMW's file",
                                                  "b: largest gap to our stored scores",
                                                  "c: largest gap to our stored scores")]
            summary.append(f"   largest gaps: a {worst[0]:.1e}, b {worst[1]:.1e}, c {worst[2]:.1e}")
            for why, group in per_tile.groupby("why", sort=False):     # each kind of tile on its own
                summary.append(f"   {why}: " + ", ".join(
                    f"{test} {int(group[f'{test} ok'].sum())} of {len(group)}" for test in "abc"))
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test whether the last day explains every gap with AMW.")
    ap.add_argument("--run", required=True, type=Path)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--quarters", nargs="+", help="e.g. 2026Q1 2026Q3")
    which.add_argument("--year", type=int, help="every quarter of the year that compare.py has results for")
    ap.add_argument("--tiles", type=int, default=4,
                    help="tiles with disagreements re-scored per subregion (default 4; 0 = no re-scoring)")
    ap.add_argument("--outside", type=int, default=0,
                    help="also re-score, per subregion, the N tiles with the most chips that disagree "
                         "away from any last-day scene (default 0)")
    ap.add_argument("--subregions", type=int, nargs="+", help="re-score only these (default: all)")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    quarters = args.quarters or [q for q in compare.quarters_of_year(run, args.year)
                                 if (run / "amw_comparison" / f"chips_{q}.parquet").exists()]
    if not quarters:
        raise SystemExit(f"LAST DAY ERROR: no compare.py results for {args.year} in "
                         f"{run / 'amw_comparison'}; run compare.py first")
    import last_day
    last_day.login(run)
    rescorer = Rescorer(run) if args.tiles > 0 or args.outside > 0 else None
    for q in quarters:
        print("\n".join(check_quarter(run, q, args.tiles, args.subregions, rescorer=rescorer,
                                         outside_per_subregion=args.outside)),
              "\n", flush=True)
    print(f"results -> {run / 'amw_comparison' / 'last_day_check'}")
