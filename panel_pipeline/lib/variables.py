"""What every column of the panel means: one place, read by the notebooks.

Each entry was written from the code that makes the column (plan.py and grid.py for the
chip's fixed facts, detect.py for the scores and quality columns, assemble_period.py for
the flags and distances), not from older notes. Line references are given so a reader can
check them.

    from variables import dictionary
    dictionary()                          # every column, in the panel's order
    dictionary(["confidence", "n_clear"]) # only these
"""
from __future__ import annotations

import pandas as pd

# (column, where it is stored, type, meaning, missing when)
# "where": plan = plan/chips.parquet (one row per chip, never changes over time);
#          scores / postprocess = the two layers of each period of the panel.
_ROWS = [
    # ---- the chip: identity and place (fixed over time) -------------------------------
    ("chip_id", "every file", "integer",
     "The chip's number: zone | tile column | tile row | chip in tile (grid.chip_ids). "
     "The same chip has the same chip_id in every period, frequency and run, so it is the "
     "panel's cross-section identifier.",
     "never"),
    ("period", "added on reading", "text",
     "The period the row belongs to: 2025-01 for a month, 2025Q1 for a quarter "
     "(the time identifier; read_panel adds it, the files keep it in the folder name).",
     "never"),
    ("tile_key", "plan", "text",
     "AMW's tile the chip belongs to (552 px tile + 12 px padding at 10 m, about 5.8 km). "
     "529 chips per tile.",
     "never"),
    ("chip_index", "plan", "integer",
     "Position of the chip inside its tile, 0 to 528 (23 x 23 chips).",
     "never"),
    ("minx, miny, maxx, maxy", "plan", "degrees",
     "The chip's square (longitude / latitude, WGS84): 48 x 48 pixels = 480 m a side. "
     "Chips overlap by half (they move 24 px at a time).",
     "never"),
    ("lon, lat", "computed here", "degrees",
     "Centre of the chip's square, (minx + maxx) / 2 and (miny + maxy) / 2.",
     "never"),
    ("amw_key_x, amw_key_y", "plan", "degrees",
     "AMW's own location key (rounded chip centre), kept only to join with AMW's published files.",
     "never"),
    ("in_region", "plan", "true / false",
     "The chip's square touches the study region. AMW scores every chip of a tile that touches "
     "the region, so chips just outside it exist too; restrict the sample with this.",
     "never"),
    ("in_andes", "plan", "true / false",
     "The chip is in AMW's Andes area, where AMW uses the lower 0.2 threshold (det_andes).",
     "never"),
    ("subregion", "scores", "integer",
     "Which of the region's subregions the chip's score came from. A tile on a border is "
     "scored by both subregions; the panel keeps the copy with the higher confidence, on a tie "
     "the lower subregion (assemble_period.place_scores).",
     "never"),
    # ---- the model's score -----------------------------------------------------------
    ("scored", "scores", "true / false",
     "The chip has a score in this period. False only where its tile failed every download "
     "try; the row is kept so 'no data' never reads as 'no mine'.",
     "never"),
    ("confidence", "scores", "0 to 1",
     "The model's mining score: the mean of the six votes, exactly AMW's number "
     "(detect.score_tile). Made from one picture per period: the median of the period's "
     "clear Sentinel-2 images.",
     "chip not scored"),
    ("vote_1 ... vote_6", "scores", "0 to 1",
     "The six ensemble members' scores behind confidence. Their spread says how much the "
     "members disagree.",
     "chip not scored"),
    # ---- how well the period was seen (quality / control variables) -------------------
    ("n_obs", "scores", "images per pixel",
     "Sentinel-2 images of the period with a cloud score, cloudy or clear, averaged over the "
     "chip's pixels. How often the satellite passed.",
     "chip not scored"),
    ("n_clear", "scores", "images per pixel",
     "Images that passed AMW's cloud test (Cloud Score+ cs_cdf >= 0.6), averaged over the "
     "chip's pixels. These are the images the median picture, and so the score, is made of.",
     "chip not scored"),
    ("n_clear_median", "scores", "images per pixel",
     "The median over the chip's pixels of the clear-image count: less pulled by a few "
     "well-seen pixels than n_clear.",
     "chip not scored"),
    ("empty_share", "scores", "0 to 1",
     "Share of the chip's pixels with no clear image at all. The model sees those pixels as "
     "black. 1 = the model saw a black square.",
     "chip not scored; or, in periods scored before 1 Nov 2026, a chip with some (not all) "
     "empty pixels (masked-pixel bug)"),
    ("thin_share", "scores", "0 to 1",
     "Share of the chip's pixels with only 1 or 2 clear images: seen, but by a thin median.",
     "chip not scored"),
    ("clear_day_mean", "scores", "days",
     "Average date of the clear images, in days since the period's first day (0 = first day). "
     "Says whether the picture is from the start or the end of the period.",
     "chip not scored, or no clear image (n_clear = 0)"),
    ("cs_mean", "scores", "0 to 1",
     "Average Cloud Score+ (cs_cdf: 1 clear, 0 cloudy) over all images of the period: how "
     "cloudy the period was over the chip.",
     "chip not scored, or no image (n_obs = 0)"),
    ("cs_clear_mean", "scores", "0 to 1",
     "Average Cloud Score+ of the clear images only: how clean the images that made the "
     "picture were (always >= 0.6).",
     "chip not scored, no clear image; or the masked-pixel bug (see empty_share)"),
    ("ndvi_mean", "scores", "-1 to 1",
     "NDVI of the median picture, (B8 - B4) / (B8 + B4), averaged over the pixels the model "
     "saw: greenness. Bare ground and water are low, forest high.",
     "chip not scored, nothing seen; or the masked-pixel bug"),
    ("bsi_mean", "scores", "-1 to 1",
     "Bare soil index of the median picture, ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2)), "
     "averaged over the pixels seen: high on exposed soil, as in open-pit mines.",
     "chip not scored, nothing seen; or the masked-pixel bug"),
    ("cirrus_mean", "scores", "reflectance",
     "Band B10 (cirrus) of the median picture / 10,000, averaged over the pixels seen: thin "
     "high cloud or haze left in the picture.",
     "chip not scored, nothing seen; or the masked-pixel bug"),
    # ---- AMW's detection rules (postprocess layer) ------------------------------------
    ("det_raw", "postprocess", "true / false",
     "AMW's raw detection: confidence > 0.4.",
     "chip not scored"),
    ("det_andes", "postprocess", "true / false",
     "AMW's Andes detection: chip in the Andes area and confidence > 0.2.",
     "chip not scored"),
    ("det_loose", "postprocess", "true / false",
     "AMW's loose rule: confidence >= 0.43, and if the chip is isolated (its 5th nearest "
     "other chip with >= 0.43 is more than 3 km away) confidence >= 0.75.",
     "chip not scored"),
    ("det_stringent", "postprocess", "true / false",
     "AMW's stringent rule: as loose with 0.55 and, if isolated, 0.8. "
     "Stringent detections are a subset of loose, loose of raw.",
     "chip not scored"),
    ("kth_km_loose", "postprocess", "km",
     "For chips with confidence >= 0.43: distance to the 5th nearest other such chip, measured "
     "AMW's way (one flat UTM map). The isolation test of det_loose uses this.",
     "chip below 0.43 (not in the catalogue), the lower-scored of two chips sharing AMW's location key, or fewer than 6 chips in the catalogue"),
    ("kth_km_geo_loose", "postprocess", "km",
     "The same distance measured on the Earth's real shape; AMW's flat map stretches it by up "
     "to ~5% at the region's ends (open decision D3).",
     "as kth_km_loose"),
    ("kth_km_stringent", "postprocess", "km",
     "As kth_km_loose, for chips with confidence >= 0.55 (the stringent catalogue).",
     "chip below 0.55, or fewer than 6 chips in the catalogue"),
    ("kth_km_geo_stringent", "postprocess", "km",
     "As kth_km_geo_loose, for the stringent catalogue.",
     "as kth_km_stringent"),
]


# Variables the analysis notebooks derive (notebooks/build_tables.py and lib/analysis_tables.py).
_DERIVED = [
    ("vote_sd", "derived", "0 to 0.5",
     "Standard deviation of the six votes (vote_1 ... vote_6) of one chip in one period: how much "
     "the six models of the ensemble disagree. 0 = all six give the same score.",
     "chip not scored"),
    ("dist_km", "derived", "km",
     "Distance from the chip's centre to the centre of the nearest known-mine chip, measured along "
     "the Earth's surface. A known-mine chip itself has 0. Which detections count as known mines is "
     "set in notebooks/analysis.yaml and printed at the top of each notebook.",
     "no known mines are set"),
    ("known mine", "derived", "true / false",
     "The chip was detected (loose rule) in one of the known-mine years: AMW's yearly files for "
     "2018 to 2024 for now, to be replaced year by year by our own quarters.",
     "never"),
    ("followed chip", "derived", "true / false",
     "A chip whose full row is kept in every period: it scored at least 0.2 in some period, is in "
     "one of AMW's files, or is a known mine. Persistence and drill-downs use these chips; counts "
     "and histograms use every chip.",
     "never"),
    ("detected / missed", "derived", "D / M / ·",
     "For one chip and the chosen rule: D = detected in the period, M = scored but not detected, "
     "· = not scored (no data, which is neither).",
     "never"),
    ("gap", "derived", "periods",
     "A run of M after a D. Closed: the chip was detected again after it (D M M D). Open: not "
     "detected again up to the last period (D M M), which may mean the mine is gone or is still "
     "being missed. Its length is the number of M in the run.",
     "never"),
    ("n_clear band", "derived", "5 groups",
     "n_clear grouped: 0 | above 0, under 1 | 1 to under 3 | 3 to under 10 | 10 or more.",
     "chip not scored"),
    ("share_n_clear_0", "derived", "0 to 1",
     "Share of a tile's (or map cell's) scored chips with n_clear = 0: the model saw only black.",
     "no chip scored"),
    ("share_n_clear_under_1", "derived", "0 to 1",
     "Share of scored chips with n_clear under 1: on average less than one clear image per pixel.",
     "no chip scored"),
    ("detections per 1,000 chips", "derived", "count",
     "Chips detected by the chosen rule per 1,000 scored chips: a rate, so areas and periods with "
     "different numbers of chips can be compared.",
     "no chip scored"),
    ("AUC", "derived", "0.5 to 1",
     "How well a cloud measure tells missed periods from detected periods of the same mines: the "
     "probability that, of a randomly picked missed and detected period, the missed one had the "
     "worse view. 0.5 = no better than a coin, 1 = perfect separation.",
     "fewer than one missed and one detected period"),
    ("AMW yearly status", "derived", "category",
     "A chip under AMW's persistence rule (recipe A, core/persistence.py) on AMW's yearly pictures: "
     "confirmed = detected in a year and in the next one; provisional = detected only in AMW's latest "
     "year at the stricter stringent rule (waiting for the next year); otherwise not in the record.",
     "chip in none of AMW's yearly files"),
    ("quarterly onset", "derived", "quarter",
     "The first quarter in which we detected the chip and detected it again within the next four "
     "quarters: the quarterly counterpart of AMW's one-year window.",
     "no such pair of detections"),
    ("amw_confidence", "AMW's files", "0 to 1",
     "AMW's own score for a chip in one of their files (yearly 2018 to 2025, quarterly from 2025). "
     "Their files only list chips scoring 0.4 or more, so every listed chip counts as an AMW "
     "detection under the raw rule; the loose and stringent rules are applied to their scores the "
     "same way as to ours.",
     "chip not in that AMW file"),
    ("source", "derived", "category",
     "Where a detection comes from: our months, our quarters, AMW's quarters or AMW's yearly "
     "pictures. Each source is one picture per period, made from all images of that period.",
     "never"),
    ("unique chips", "derived", "count",
     "Chips detected at least once in a set of periods, each chip counted once however often it "
     "was detected. Cumulative unique chips grow as periods are added; they never go down.",
     "never"),
    ("quarter only / month only / both", "derived", "category",
     "For a quarter and one of its months: quarter only = detected in the quarter's picture but not "
     "in the month's; month only = the reverse; both = in both pictures.",
     "chip not scored in one of the two"),
    ("months that caught it", "derived", "0 to 3",
     "For a chip detected in a quarter: in how many of that quarter's three months it was also "
     "detected.",
     "a month of the quarter not assembled"),
]


def describe(names: list[str]) -> list[dict]:
    """[{variable, unit, meaning, missing}] for these names, panel columns and derived ones alike."""
    rows = {}
    for name, _where, kind, meaning, missing in _ROWS + _DERIVED:
        for n in [name, *expand(name)]:
            rows[n] = {"variable": n, "unit": kind, "meaning": meaning, "missing": missing}
    return [rows.get(n, {"variable": n, "unit": "?", "meaning": "not described in variables.py", "missing": "?"})
            for n in names]


def expand(name: str) -> list[str]:
    """'vote_1 ... vote_6' -> vote_1..vote_6; 'a, b' -> a, b."""
    if name == "vote_1 ... vote_6":
        return [f"vote_{i}" for i in range(1, 7)]
    return [n.strip() for n in name.split(",")]


def dictionary(columns: list[str] | None = None, grouped: bool = True) -> pd.DataFrame:
    """The variables as a table: variable, stored in, type / unit, meaning, missing when.

    columns: keep only these, in this order (names not described are listed as such).
    grouped: one row for vote_1..vote_6 and for the four square corners; False = one per column.
    """
    rows = []
    for name, where, kind, meaning, missing in _ROWS:
        names = [name] if grouped else expand(name)
        for n in names:
            rows.append({"variable": n, "stored in": where, "type / unit": kind,
                         "meaning": meaning, "missing when": missing})
    table = pd.DataFrame(rows)
    if columns is None:
        return table
    known = {n: r for r in table.to_dict("records") for n in expand(r["variable"])}
    out = []
    for c in columns:
        r = known.get(c, {"variable": c, "stored in": "?", "type / unit": "?",
                          "meaning": "not described in panel_pipeline/lib/variables.py",
                          "missing when": "?"})
        out.append(r | {"variable": c})
    return pd.DataFrame(out)
