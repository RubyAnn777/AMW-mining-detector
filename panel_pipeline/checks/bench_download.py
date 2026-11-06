"""Measure what one tile costs Earth Engine, and whether a cheaper recipe gives the same pixels.

    python panel_pipeline/checks/bench_download.py --run $RUN --period 2026Q3
    python panel_pipeline/checks/bench_download.py --run $RUN --period 2026-09 --tiles 5
    ... --compare bounded      test the filterBounds recipe instead (see below)
    ... --amw                  also download AMW's recipe alone, on other tiles

Downloads a few tiles of a planned run with two recipes, one request at a time, and asks
Earth Engine for each request's compute profile (ee.data.profiling: the work its servers
did, in EECU-seconds, which is what a project's monthly quota counts):
    panel     what detect.py downloads: AMW's recipe plus the five quality bands
              (22 Oct: ~130 EECU-s per tile-quarter)
    selected  (default --compare) the same, but the images keep only the 13 bands the
              model reads before the median. AMW's recipe takes the median of every band
              of the collection (QA and classification bands, the joined cs_cdf) and the
              request picks the 13 afterwards (gee.py, _build_composite and get_tile_data).
              Each band's median is computed on its own, so the pixels must be identical;
              the question is whether Earth Engine already skips the unused bands
    bounded   (--compare bounded) the same as panel, but every image collection is first
              cut to the tile's footprint (filterBounds). On 22 Oct (5 tiles of subregion 3,
              2026Q3) its pixels equalled panel's, but it saved no work reliably
    amw       (--amw) AMW's recipe only: the 13 bands' cloud-free median, on other tiles,
              so no request can reuse another's results
Each tile is downloaded with both recipes (taking turns going first), and the two are
compared value by value. A recipe is only safe for the panel if every tile says
"identical".

Also prints how many MB each answer was. Sends 2 x --tiles requests (3 x with --amw) to
the run's Earth Engine project, about 50-200 EECU-s each; writes nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from descarteslabs.geo import DLTile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # detect.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))   # our helpers: config, periods, ...

import detect
from config import load_config
from periods import Period

import gee


class Bounded(detect.PanelExtractor):
    """detect.py's recipe, with each collection cut to the tile's footprint first."""

    region = None                                # the tile being downloaded

    def _build_composite(self, start_date, end_date):
        # Copy of AMW's recipe (gee.py:208-229) and detect.py's extra bands, with one change:
        # .filterBounds(self.region) after .filterDate.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(start_date, end_date)
        if self.region is not None:
            s2 = s2.filterBounds(self.region)
        cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        t = self.config.clear_threshold
        composite = (s2.linkCollection(cs, ["cs_cdf"])
                     .map(lambda img: img.updateMask(img.select("cs_cdf").gte(t)))
                     .median().toFloat())
        return composite.addBands(detect.quality_bands(s2, start_date, t))

    def get_tile_data(self, tile):
        self.region = ee.Geometry.Rectangle(tile.geometry.bounds)
        self.composite = self._build_composite(self.start_date, self.end_date)
        return super().get_tile_data(tile)


RECEIVED = {}                                    # the size of the last answer
_compute_pixels = ee.data.computePixels


class Selected(detect.PanelExtractor):
    """detect.py's recipe, with only the 13 model bands kept before the median."""

    def _build_composite(self, start_date, end_date):
        # Copy of AMW's recipe (gee.py, _build_composite) and detect.py's extra bands, with
        # one change: .select(SPECTRAL) before the join and again before the median, so
        # the median runs on the 13 bands that are downloaded, not on every band.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(start_date, end_date)
        cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        t = self.config.clear_threshold
        composite = (s2.select(detect.SPECTRAL)
                     .linkCollection(cs, ["cs_cdf"])
                     .map(lambda img: img.updateMask(img.select("cs_cdf").gte(t)))
                     .select(detect.SPECTRAL)
                     .median().toFloat())
        return composite.addBands(detect.quality_bands(s2, start_date, t))


RECIPES = {"selected": Selected, "bounded": Bounded}


def measured_compute_pixels(params, *args, **kwargs):
    """ee.data.computePixels, noting the size of what Earth Engine sent back."""
    pixels = _compute_pixels(params, *args, **kwargs)
    RECEIVED["mb"] = pixels.nbytes / 1e6
    return pixels


ee.data.computePixels = measured_compute_pixels  # gee.py looks it up at every call


def profiled(extractor, tile) -> tuple[np.ndarray, float, str]:
    """Download one tile; its pixels, seconds, and Earth Engine's profile of the request.

    Earth Engine sometimes answers "Profile not found" for a request it did answer (seen
    on 22 Oct); ee.profilePrinting (ee/_helpers.py) retries for the same reason. If it
    still has none after three tries, the profile is "".
    """
    ids = []
    started = time.perf_counter()
    with ee.data.profiling(ids.append):
        pixels = extractor.get_tile_data(tile)
    seconds = time.perf_counter() - started
    get_profiles = ee.ApiFunction.lookup("Profile.getProfiles").call
    for _ in range(3):
        try:
            return pixels, seconds, get_profiles(ids=ids).getInfo() if ids else ""
        except ee.EEException as e:
            if "not found" not in str(e):
                raise
            time.sleep(5)
    return pixels, seconds, ""


def eecu_seconds(profile: str) -> float:
    """The EECU-seconds column of a profile table, summed (nan if there is no such column)."""
    lines = profile.strip().splitlines()
    head = next((i for i, line in enumerate(lines) if "EECU" in line), None)
    if head is None:
        return float("nan")
    column = lines[head].split().index(next(w for w in lines[head].split() if "EECU" in w))
    total = 0.0
    for line in lines[head + 1:]:
        words = line.split()
        try:
            total += float(words[column])
        except (IndexError, ValueError):
            pass
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure what one tile costs Earth Engine.")
    ap.add_argument("--run", default="tests_output/amw_compare_s3_2026-10-19")
    ap.add_argument("--period", default="2026Q3", help="e.g. 2026Q3 or 2026-09")
    ap.add_argument("--tiles", type=int, default=3, help="tiles per recipe (default 3)")
    ap.add_argument("--start", type=int, default=30000,
                    help="first tile, by its row in plan/tiles.csv (default 30000)")
    ap.add_argument("--compare", choices=sorted(RECIPES), default="selected",
                    help="the recipe to test against panel (default: selected)")
    ap.add_argument("--amw", action="store_true",
                    help="also download AMW's recipe alone, on other tiles")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    cfg = load_config(run_dir / "run" / "config.yaml")
    period = Period.parse(args.period)
    keys = pd.read_csv(run_dir / "plan" / "tiles.csv")["tile_key"]
    tiles = [DLTile.from_key(k) for k in keys[args.start:args.start + 2 * args.tiles]]

    gee.EE_PROJECT = cfg.ee_project
    gee._ensure_earth_engine_initialized()
    ee.data.setDeadline(detect.REQUEST_TIMEOUT_S * 1000)
    panel = detect.extractor_for(cfg, period, None)
    test = RECIPES[args.compare](period.start.isoformat(), period.end.isoformat(),
                                 panel.config)
    amw_config = gee.DataConfig(tilesize=cfg.imagery.tilesize, pad=cfg.imagery.pad,
                                collection=cfg.imagery.collection, bands=detect.SPECTRAL,
                                clear_threshold=cfg.imagery.clear_threshold,
                                max_workers=1, image_cache_dir=None)
    amw = gee.GEE_Data_Extractor(period.start.isoformat(), period.end.isoformat(),
                                 amw_config)

    print(f"{args.period}, {args.compare} vs panel, tiles {args.start} to "
          f"{args.start + args.tiles - 1} of {run_dir.name}, one request at a time\n",
          flush=True)
    rows, verdicts = [], []
    pairs = ((args.compare, test), ("panel", panel))
    for i, (tile, other) in enumerate(zip(tiles[:args.tiles], tiles[args.tiles:])):
        # the two recipes take turns going first, in case the second one of a tile
        # profits from work Earth Engine kept from the first
        order = pairs if i % 2 == 0 else pairs[::-1]
        extra = (("amw", amw, other),) if args.amw else ()
        for name, extractor, t in (*((n, e, tile) for n, e in order), *extra):
            pixels, seconds, profile = profiled(extractor, t)
            rows.append({"recipe": name, "tile": t.key, "seconds": seconds,
                         "eecu_s": eecu_seconds(profile), "mb": RECEIVED["mb"],
                         "pixels": pixels, "profile": profile})
            print(f"  {name:<8} {t.key:<26} {seconds:6.1f} s, "
                  + (f"{rows[-1]['eecu_s']:8.1f} EECU-s" if profile else "  no profile")
                  + f", {RECEIVED['mb']:5.1f} MB of pixels", flush=True)
            if name == order[1][0]:                  # both recipes have this tile now
                a, b = rows[-2]["pixels"], rows[-1]["pixels"]     # (H, W, bands) float32
                # identical: same values, and the same places masked (-inf) or missing
                same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
                verdicts.append(same)
                differ = ~((a == b) | (np.isnan(a) & np.isnan(b))) if a.shape == b.shape else None
                print(f"  {args.compare} vs panel pixels: "
                      + ("identical" if same else "DIFFERENT")
                      + ("" if same or differ is None else
                         f", {differ.mean():.2%} of values differ, in bands "
                         + ", ".join(n for n, d in zip(detect.SPECTRAL + detect.EXTRA,
                                                        differ.any(axis=(0, 1))) if d)),
                      flush=True)
        print(flush=True)

    table = pd.DataFrame(rows).drop(columns=["pixels", "profile"])
    print("Average per tile (a recipe's tiles without a profile are left out):\n"
          + table.groupby("recipe")[["seconds", "eecu_s", "mb"]].mean().round(1).to_string())

    # The same tile with both recipes is the fair comparison: tiles differ in cost far more
    # than the recipes do (56-213 EECU-s on 22 Oct), so averages over different tiles mislead.
    pair = (table[table["recipe"].isin(["panel", args.compare])]
            .pivot(index="tile", columns="recipe", values="eecu_s").dropna())
    if len(pair):
        change = pair[args.compare] / pair["panel"] - 1
        print(f"\nSame tile, both profiled ({len(pair)} tiles): {args.compare} vs panel "
              + ", ".join(f"{c:+.0%}" for c in change)
              + f"; median {change.median():+.0%} (negative = {args.compare} is cheaper)")
    print(f"Pixels identical on {sum(verdicts)} of {len(verdicts)} tiles")
    if len(pair):
        tile = pair.index[0]
        for name in ("panel", args.compare):
            profile = next(r["profile"] for r in rows if r["tile"] == tile and r["recipe"] == name)
            print(f"\nWhere {name}'s EECU-seconds go on tile {tile} (Earth Engine's profile):")
            print("\n".join(profile.strip().splitlines()[:15]))
