"""Test-set metrics for confirmation recipes A-D, on the persistence_evaluation protocol.

Chip is predicted positive iff it intersects any patch in the layer. Pooled over
val + test2 + test3, matching model_evaluation.ipynb section 3.
"""
# >> FILE OVERVIEW
# >> ONE-OFF ANALYSIS SCRIPT, not part of the production pipeline (nothing imports it).
# >> Scores confirmation recipes A-D (see persistence.py) against labelled test chips:
# >> precision / recall / F1 per recipe. Evidence behind choosing Recipe A.
# >> Hard-coded paths under /home/zu/... (the upstream author's machine): will NOT run
# >> here without editing ROOT and the sys.path line. Upstream file.
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from sklearn.metrics import confusion_matrix

# >> Hard-coded import path to the upstream author's checkout (/home/zu).
sys.path.insert(0, "/home/zu/Genome/mining-detector/core")
from persistence import Period, PersistenceConfig, compute_onsets
from postprocess import centroid_key

# >> Hard-coded data locations: training chips folder and model output folder.
ROOT = Path("/home/zu/Genome/mining-detector")
DATA = ROOT / "data/training_patches2026-05-04T09:47"
M = ROOT / "data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble"
# >> Raw-detections stem: model name + raw floor 0.40.
STEM = "Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40"
# >> Labelled splits pooled for the metric (validation + two test sets).
SPLITS = ["val", "test2", "test3"]
# >> Annual periods 2018-2025 and quarters Q1 2025 .. Q2 2026.
YEARS = list(range(2018, 2026))
QUARTERS = ["Q125", "Q225", "Q325", "Q425", "Q126", "Q226"]


# >> Loads labelled chips (GeoTIFFs in <region>/<split>/<0|1>/) as footprint boxes.
# >> Out: GeoDataFrame with y = 0 (not mine) / 1 (mine), EPSG:4326 boxes from tif bounds.
def load_chips():
    geoms, labels = [], []
    for split in SPLITS:
        for lab in (0, 1):
            # >> Loop: split -> label -> files sorted by name (defines row order).
            for fp in sorted(DATA.glob(f"*/{split}/{lab}/*.tif")):
                with rasterio.open(fp) as src:
                    # >> rasterio reads only the header to get the chip's bounding box.
                    geoms.append(box(*src.bounds))
                labels.append(lab)
    return gpd.GeoDataFrame({"y": labels}, geometry=geoms, crs="EPSG:4326")


# >> Reads one postprocessed file for a period and threshold pair.
# >> Out: list of location keys (rounded centres) and list of chip geometries.
def load(period, t_main, t_iso):
    tag = f"t{t_main:g}_d5_3km_t-iso{t_iso:g}"
    p = M / f"postprocessed_{tag}" / f"{STEM}_{period.date_span}_{tag}.geojson"
    g = gpd.read_file(p, columns=["confidence"])
    kx, ky = centroid_key(g)
    return list(zip(kx, ky)), list(g.geometry)


# >> Builds each recipe's detection layer and scores it against labelled chips.
# >> Writes rule_metrics.csv next to this script (core/rule_metrics.csv).
def main():
    chips = load_chips()
    print(f"{len(chips):,} chips, {int(chips.y.sum()):,} positive")

    geom_of, det43, det55 = {}, {}, {}
    # >> Loop over all periods: LOOSE (0.43/0.75) always; STRINGENT (0.55/0.8) if annual.
    for p in [Period(y) for y in YEARS] + [Period.parse(t) for t in QUARTERS]:
        keys, geoms = load(p, 0.43, 0.75)
        det43[p] = set(keys)
        # >> Remember one geometry per location key, first one seen wins.
        for k, g in zip(keys, geoms):
            geom_of.setdefault(k, g)
        if p.is_annual:
            k55, g55 = load(p, 0.55, 0.8)
            det55[p] = set(k55)
            for k, g in zip(k55, g55):
                geom_of.setdefault(k, g)

    annual = [Period(y) for y in YEARS]
    # >> The five recipe variants compared (A = the one adopted in persistence.py).
    rules = {
        "A  window2 annual": PersistenceConfig(window=2),
        "B  window2 +quarters": PersistenceConfig(window=2, use_quarterly_witnesses=True),
        "C  window3 annual (classic)": PersistenceConfig(window=3),
        "D  window3 +quarters": PersistenceConfig(window=3, use_quarterly_witnesses=True),
        "D+ window3 +quarters, early": PersistenceConfig(
            window=3, use_quarterly_witnesses=True, early_confirm=True),
    }

    layers = {}
    # >> Reference layers: plain unions with no persistence rule.
    # references
    layers["or_t055_2025  (current production)"] = set().union(*det55.values())
    layers["or_t043_2025  (loose, no persistence)"] = set().union(
        *[det43[p] for p in annual])

    # >> Loop per recipe: confirmed onsets + provisional t0.55 chips for unclosed years.
    for name, cfg in rules.items():
        detected = det43 if cfg.use_quarterly_witnesses else {p: det43[p] for p in annual}
        # >> Same rule function production uses (persistence.py:compute_onsets).
        onsets = compute_onsets(detected, cfg)
        confirmed = set(onsets)
        # provisional t0.55 for annual periods whose window has not closed
        prov = set()
        for p in annual:
            if not cfg.window_is_closed(p, annual):
                prov |= det55[p]
        layers[f"{name}"] = confirmed | (prov - confirmed)
        layers[f"{name}  [confirmed only]"] = confirmed

    # >> Scoring: a labelled chip counts as predicted mine if any layer chip touches it.
    rows = []
    for name, keys in layers.items():
        gdf = gpd.GeoDataFrame(geometry=[geom_of[k] for k in keys if k in geom_of],
                               crs="EPSG:4326")
        # >> Spatial join: labelled chips x detection chips that intersect.
        hits = gpd.sjoin(chips[["geometry"]], gdf[["geometry"]],
                         how="inner", predicate="intersects")
        pred = chips.index.isin(hits.index.unique()).astype(int)
        # >> sklearn confusion matrix -> TN, FP, FN, TP counts.
        tn, fp, fn, tp = confusion_matrix(chips["y"], pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(layer=name, patches=len(keys), TP=tp, FP=fp, FN=fn, TN=tn,
                         precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    # >> File write: metrics table as CSV.
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).parent / "rule_metrics.csv", index=False)


if __name__ == "__main__":
    main()
