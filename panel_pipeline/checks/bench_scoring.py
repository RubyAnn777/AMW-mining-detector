"""Time the scoring of a few kept tiles, with and without AMW's macOS TensorFlow setting.

    python panel_pipeline/checks/bench_scoring.py
    python panel_pipeline/checks/bench_scoring.py --tiles 4 --dates 2026-07-01_2026-10-01

Scores the same tiles three ways, each in a fresh Python (the setting can't be undone
inside one):
    amw     with AMW's macOS setting (core/tf_darwin.py): how detect.py scores on a Mac now
    graph   without it: TensorFlow compiles the model first, as on Linux (AMW's VMs, the
            cluster). AMW added the setting against crashes on Macs, so this may crash
    direct  with AMW's setting, but all of a tile's chips go through the model in one go,
            instead of model.predict's batches of 32
Then prints the seconds per tile of each, and how far its scores are from "amw".

Reads tiles only from an image cache (detect.py --image-cache), so no Earth Engine
requests. Writes nothing in the repository: the scores go to a temporary folder.
While detect.py runs on the same computer, both share its processors, so every time
here is slower than it would be alone; the three are run one after the other, so they
are slowed alike.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))   # our helpers: config, periods, ...

from config import REPO, load_config

MODES = ("amw", "graph", "direct")


def cached_tiles(cache: Path, dates: str, n: int) -> list[Path]:
    """The first n kept images of these dates (file names: S2L1C_<tile key>_<dates>.tif)."""
    files = sorted(cache.glob(f"S2L1C_*_{dates}.tif"))
    if len(files) < n:
        raise SystemExit(f"BENCH ERROR: {cache} has {len(files)} images of {dates}, "
                         f"{n} needed")
    return files[:n]


class Direct:
    """The model, called on all chips at once; looks like a model to detect.score_tile."""

    def __init__(self, model):
        self.model = model

    def predict(self, chips, verbose=0):
        return self.model(chips, training=False).numpy()


def score(mode: str, run_dir: Path, files: list[Path], out: Path) -> None:
    """One mode, in this Python: score the tiles, save the scores, print the times."""
    import tensorflow as tf

    import detect                                # puts AMW's core/ on the search path
    import gee
    from descarteslabs.geo import DLTile
    from tf_darwin import apply_darwin_tf_compat

    cfg = load_config(run_dir / "run" / "config.yaml")
    if mode != "graph":
        apply_darwin_tf_compat()                 # what detect.py does (on macOS only)
    model = tf.keras.models.load_model(str(cfg.inference.model_path), compile=False)
    if mode == "direct":
        model = Direct(model)
    stride = detect.CHIP_PX // cfg.inference.stride_ratio

    frames, seconds = [], []
    for i, path in enumerate(files):
        key = path.name.removeprefix("S2L1C_").rsplit("_", 2)[0]
        tile = DLTile.from_key(key)                       # no network
        pixels = gee.GEE_Data_Extractor.load_tile(None, path)   # AMW's cache reader
        started = time.perf_counter()
        rows = detect.score_tile(tile, pixels, model, stride)
        if i > 0:              # the first tile also builds the model's machinery: not timed
            seconds.append(time.perf_counter() - started)
            frames.append(rows)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print(json.dumps({"seconds": seconds}))


def compare(ref: pd.DataFrame, other: pd.DataFrame, threshold: float) -> str:
    """How far one mode's scores are from the reference's, chip by chip."""
    both = ref.merge(other, on="chip_id", suffixes=("", "_x"))
    if len(both) != len(ref):
        return "DIFFERENT CHIPS"
    diff = np.abs(both["confidence"] - both["confidence_x"]).max()
    votes = max(np.abs(both[c] - both[f"{c}_x"]).max()
                for c in ref.columns if c.startswith("vote_"))
    flips = ((both["confidence"] > threshold) != (both["confidence_x"] > threshold)).sum()
    same = "identical" if diff == 0 and votes == 0 else "differ"
    return (f"{same}: largest confidence difference {diff:.2e}, largest vote difference "
            f"{votes:.2e}, {flips} chip(s) on the other side of {threshold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Time scoring with and without AMW's "
                                             "macOS TensorFlow setting.")
    ap.add_argument("--run", default="tests_output/amw_compare_s3_2026-10-19",
                    help="a planned run, for its model and stride")
    ap.add_argument("--cache", default="tests_output/image_cache",
                    help="the image cache to read tiles from")
    ap.add_argument("--dates", default="2026-07-01_2026-10-01",
                    help="the kept images' dates, as in their file names (default: 2026Q3)")
    ap.add_argument("--tiles", type=int, default=6, help="tiles to time (default 6)")
    ap.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)   # one child run
    ap.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    run_dir = (REPO / args.run).resolve()
    files = cached_tiles((REPO / args.cache).resolve(), args.dates, args.tiles + 1)
    if args.mode:
        score(args.mode, run_dir, files, args.out)
        sys.exit()

    threshold = load_config(run_dir / "run" / "config.yaml").thresholds.raw
    print(f"Scoring {args.tiles} tiles of {args.dates} three ways (plus one warm-up tile "
          f"each, not timed)", flush=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in MODES:
            out = Path(tmp) / f"{mode}.parquet"
            child = subprocess.run(
                [sys.executable, __file__, "--run", args.run, "--cache", args.cache,
                 "--dates", args.dates, "--tiles", str(args.tiles), "--mode", mode,
                 "--out", str(out)],
                capture_output=True, text=True, env=os.environ | {"TF_CPP_MIN_LOG_LEVEL": "2"})
            if child.returncode != 0:
                tail = "\n      ".join(child.stderr.strip().splitlines()[-5:])
                print(f"  {mode:<7} FAILED (exit {child.returncode}):\n      {tail}", flush=True)
                continue
            seconds = json.loads(child.stdout.strip().splitlines()[-1])["seconds"]
            results[mode] = pd.read_parquet(out)
            line = f"  {mode:<7} {np.mean(seconds):6.2f} s per tile"
            if mode != "amw" and "amw" in results:
                line += "   scores " + compare(results["amw"], results[mode], threshold)
            print(line, flush=True)
