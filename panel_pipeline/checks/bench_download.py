"""Measure what one tile costs Earth Engine, and whether a cheaper recipe gives the same pixels.

    python panel_pipeline/checks/bench_download.py
    python panel_pipeline/checks/bench_download.py --tiles 3 --period 2026Q3 --start 30000

Downloads a few tiles of a planned run three ways, one request at a time, and asks Earth
Engine for each request's compute profile (ee.data.profiling: the work its servers did,
in EECU-seconds, which is what a project's monthly quota counts):
    amw       AMW's recipe only: the 13 bands' cloud-free median (gee.py:208-229)
    panel     what detect.py downloads: AMW's recipe plus the two image counts
    bounded   the same as panel, but every image collection is first cut to the tile's
              footprint (filterBounds), so Earth Engine only joins and scans the images
              that touch the tile. On 22 Oct (5 tiles of subregion 3, 2026Q3) its pixels
              equalled panel's, but it saved no work reliably: Earth Engine already loads
              only the images that touch a tile, and the median is most of the cost
Each "bounded" tile is downloaded again as "panel", and the two are compared pixel by
pixel. "amw" uses other tiles, so no request can reuse another's results.

Also prints how many MB each answer was: at a given internet speed, the time a download
needs grows with it. Sends 3 x --tiles requests to the run's Earth Engine project; writes nothing.
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
        # Copy of AMW's recipe (gee.py:208-229) and detect.py's counts, with one change:
        # .filterBounds(self.region) after .filterDate.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(start_date, end_date)
        if self.region is not None:
            s2 = s2.filterBounds(self.region)
        cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        t = self.config.clear_threshold
        composite = (s2.linkCollection(cs, ["cs_cdf"])
                     .map(lambda img: img.updateMask(img.select("cs_cdf").gte(t)))
                     .median().toFloat())
        scores = s2.linkCollection(cs, ["cs_cdf"]).select("cs_cdf")
        clear = scores.map(lambda img: img.updateMask(img.gte(t)))
        return (composite.addBands(scores.count().rename("n_obs").toUint16())
                         .addBands(clear.count().rename("n_clear").toUint16()))

    def get_tile_data(self, tile):
        self.region = ee.Geometry.Rectangle(tile.geometry.bounds)
        self.composite = self._build_composite(self.start_date, self.end_date)
        return super().get_tile_data(tile)


RECEIVED = {}                                    # the size of the last answer
_compute_pixels = ee.data.computePixels


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
    ap.add_argument("--period", default="2026Q3")
    ap.add_argument("--tiles", type=int, default=3, help="tiles per recipe (default 3)")
    ap.add_argument("--start", type=int, default=30000,
                    help="first tile, by its row in plan/tiles.csv (default 30000)")
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
    bounded = Bounded(period.start.isoformat(), period.end.isoformat(), panel.config)
    amw_config = gee.DataConfig(tilesize=cfg.imagery.tilesize, pad=cfg.imagery.pad,
                                collection=cfg.imagery.collection, bands=detect.SPECTRAL,
                                clear_threshold=cfg.imagery.clear_threshold,
                                max_workers=1, image_cache_dir=None)
    amw = gee.GEE_Data_Extractor(period.start.isoformat(), period.end.isoformat(),
                                 amw_config)

    print(f"{args.period}, tiles {args.start} to {args.start + 2 * args.tiles - 1} of "
          f"{run_dir.name}, one request at a time\n", flush=True)
    rows, profiles = [], {}
    pairs = (("bounded", bounded), ("panel", panel))
    for i, (tile, other) in enumerate(zip(tiles[:args.tiles], tiles[args.tiles:])):
        # bounded and panel take turns going first, in case the second one of a tile
        # profits from work Earth Engine kept from the first
        order = pairs if i % 2 == 0 else pairs[::-1]
        for name, extractor, t in (*((n, e, tile) for n, e in order), ("amw", amw, other)):
            pixels, seconds, profile = profiled(extractor, t)
            if profile:
                profiles.setdefault(name, profile)
            rows.append({"recipe": name, "tile": t.key, "seconds": seconds,
                         "eecu_s": eecu_seconds(profile), "mb": RECEIVED["mb"],
                         "pixels": pixels})
            print(f"  {name:<8} {t.key:<26} {seconds:6.1f} s, "
                  + (f"{rows[-1]['eecu_s']:8.1f} EECU-s" if profile else "  no profile")
                  + f", {RECEIVED['mb']:5.1f} MB of pixels", flush=True)
            if name == order[1][0]:                  # both recipes have this tile now
                a, b = rows[-2]["pixels"], rows[-1]["pixels"]     # (H, W, bands) float32
                same = np.array_equal(a, b, equal_nan=True)
                differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
                print(f"  bounded vs panel pixels: {'identical' if same else 'DIFFERENT'}"
                      + ("" if same else
                         f", largest difference {np.nanmax(np.abs(a - b)):.4g}, "
                         f"{differ.mean():.2%} of values differ"), flush=True)
        print(flush=True)

    table = pd.DataFrame(rows).groupby("recipe")[["seconds", "eecu_s", "mb"]].mean()
    print("Average per tile:\n" + table.round(1).to_string())
    for name in ("panel", "bounded"):
        if name in profiles:
            print(f"\nWhere {name}'s EECU-seconds go (Earth Engine's profile of one request):")
            print("\n".join(profiles[name].strip().splitlines()[:15]))
