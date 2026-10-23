"""Check the Earth Engine login without downloading any imagery.

    python cluster/check_login.py                  uses config.yaml's earth_engine.project
    python cluster/check_login.py --config configs/<run>.yaml

Logs in the way detect.py does (AMW's gee.py: the service account named by
GOOGLE_APPLICATION_CREDENTIALS if set, else your own login) and asks Earth Engine one
tiny question: how many Sentinel-2 images cover a point in La Pampa in July 2024.
Prints who is logged in and the answer. Costs nothing worth counting.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "panel_pipeline"))   # config.py, grid.py (adds core/)

from config import DEFAULT_CONFIG, load_config   # noqa: E402
import grid                                       # noqa: E402,F401  (puts AMW's core/ on the path)
import ee                                         # noqa: E402
import gee                                        # noqa: E402  (AMW)

ap = argparse.ArgumentParser(description="Check the Earth Engine login.")
ap.add_argument("--config", default=str(DEFAULT_CONFIG))
args = ap.parse_args()

cfg = load_config(args.config)
key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
print(f"project     {cfg.ee_project}")
print(f"login       {'service account key ' + key if key else 'your own earthengine authenticate login'}")

gee.EE_PROJECT = cfg.ee_project
gee._ensure_earth_engine_initialized()            # the same call detect.py makes first
n = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
       .filterBounds(ee.Geometry.Point(-69.9, -12.9))
       .filterDate("2024-07-01", "2024-08-01").size().getInfo())
print(f"answer      {n} Sentinel-2 images over La Pampa in July 2024")
print("login works")
