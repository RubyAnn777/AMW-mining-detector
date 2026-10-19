"""Download AMW's published files (Source Cooperative).

    python amw_comparison/download_amw.py --quarters 2026Q1 2026Q2 2026Q3
    python amw_comparison/download_amw.py --panel

Saves into amw_comparison/data/amw/ (ignored by git). --quarters: amazon_basin_Q126_t0.4.geojson,
..., every chip AMW scored above 0.4 in that quarter, all six subregions merged and
deduplicated. --panel: amazon_basin_detections.geojson, one row per location with its onset
and status. Files already there are kept; delete one to fetch it again.
File names: data/outputs/MANIFEST.yaml (filename_grammar, path_map).
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request

from compare import AMW_DIR, AMW_PANEL, amw_file, sha256

ROOT = "https://data.source.coop/earthgenome/amazon-mining-watch"
BROWSE = "https://source.coop/earthgenome/amazon-mining-watch"


def fetch(url: str, path) -> None:
    """Download to a temporary name, then rename: a cut-off download leaves no file."""
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download AMW's published files.")
    ap.add_argument("--quarters", nargs="*", default=[], help="our tags, e.g. 2026Q3")
    ap.add_argument("--panel", action="store_true", help="also amazon_basin_detections.geojson")
    args = ap.parse_args()

    wanted = [(f"raw_detections/{amw_file(q).name}", amw_file(q)) for q in args.quarters]
    if args.panel:
        wanted.append((AMW_PANEL.name, AMW_PANEL))
    if not wanted:
        ap.error("name --quarters, --panel, or both")

    AMW_DIR.mkdir(parents=True, exist_ok=True)
    for key, path in wanted:
        if not path.exists():
            print(f"downloading {ROOT}/{key}", flush=True)
            try:
                fetch(f"{ROOT}/{key}", path)
            except Exception as err:
                raise SystemExit(f"DOWNLOAD ERROR: {err}\nCheck the name in the browser: {BROWSE}")
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB  sha256 {sha256(path)[:12]}")
