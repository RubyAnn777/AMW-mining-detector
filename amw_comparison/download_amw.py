"""Download AMW's published raw detections for some quarters (Source Cooperative).

    python amw_comparison/download_amw.py --quarters 2026Q1 2026Q2 2026Q3

Saves amw_comparison/data/amw/amazon_basin_Q126_t0.4.geojson, ... (ignored by git):
every chip AMW scored above 0.4 in that quarter, all six subregions merged and
deduplicated. Files already there are kept; delete one to fetch it again.
File names: data/outputs/MANIFEST.yaml (filename_grammar, path_map).
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request

from compare import AMW_DIR, amw_file, sha256

ROOT = "https://data.source.coop/earthgenome/amazon-mining-watch/raw_detections"
BROWSE = "https://source.coop/earthgenome/amazon-mining-watch/raw_detections"


def fetch(url: str, path) -> None:
    """Download to a temporary name, then rename: a cut-off download leaves no file."""
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download AMW's raw detections.")
    ap.add_argument("--quarters", nargs="+", required=True, help="our tags, e.g. 2026Q3")
    args = ap.parse_args()

    AMW_DIR.mkdir(parents=True, exist_ok=True)
    for quarter in args.quarters:
        path = amw_file(quarter)
        if not path.exists():
            print(f"downloading {ROOT}/{path.name}", flush=True)
            try:
                fetch(f"{ROOT}/{path.name}", path)
            except Exception as err:
                raise SystemExit(f"DOWNLOAD ERROR: {err}\nCheck the name in the browser: {BROWSE}")
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB  sha256 {sha256(path)[:12]}")
