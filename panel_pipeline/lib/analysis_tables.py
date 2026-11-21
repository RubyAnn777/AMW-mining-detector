"""Reading <run>/analysis/ (written by notebooks/build_tables.py) and the measures built on it.

    from analysis_tables import Analysis
    a = Analysis(run)
    a.periods("quarterly")                         # ['2025Q1', ..., '2026Q3']
    hits = a.table("quarterly", "hits")            # one row per followed chip and period
    tiles = a.table("quarterly", "tiles")          # one row per tile and period
    m = detection_matrix(hits, "det_loose")        # chip x period: 1 detected, 0 missed, NaN not scored

Everything the four analysis notebooks compute is here, in plain functions, so the
notebooks only choose and draw, and every number can be checked from a Python prompt.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from periods import Period

# ------------------------------------------------------------------ bins (the builder uses these too)

SCORE_EDGES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
               0.43, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0000001]
NCLEAR_EDGES = [0, 1e-9, 1, 3, 10, np.inf]
NCLEAR_LABELS = ["0", "above 0, under 1", "1 to under 3", "3 to under 10", "10 or more"]
EMPTY_EDGES = [0, 1e-9, 0.5, 1, 1.0000001]
EMPTY_LABELS = ["0 (every pixel seen)", "above 0, up to half", "over half, not all", "1 (all black)"]
DIST_EDGES = [0, 0.5, 1, 5, 20, 50, np.inf]
DIST_LABELS = ["under 0.5 km", "0.5 to 1 km", "1 to 5 km", "5 to 20 km", "20 to 50 km", "50 km or more"]

RULES = {"det_raw": "raw: confidence > 0.4",
         "det_loose": "loose: ≥ 0.43, or ≥ 0.75 if isolated",
         "det_stringent": "stringent: ≥ 0.55, or ≥ 0.8 if isolated",
         "det_andes": "Andes: > 0.2, Andes chips only"}
THRESHOLD = {"det_raw": 0.4, "det_loose": 0.43, "det_stringent": 0.55, "det_andes": 0.2}

# The cloud measures the notebooks compare. Direction: +1 = a higher value means a clearer view.
CLOUD_MEASURES = {
    "n_clear": ("clear images per pixel (mean over the chip)", +1),
    "n_clear_median": ("clear images per pixel (median over the chip)", +1),
    "empty_share": ("share of the chip with no clear image (black to the model)", -1),
    "thin_share": ("share of the chip with only 1 or 2 clear images", -1),
    "cs_mean": ("mean Cloud Score+ of all images (1 = clear)", +1),
    "cirrus_mean": ("haze left in the picture (band B10)", -1),
}


def score_label(i: int) -> str:
    if i < 0:
        return "not scored"
    lo, hi = SCORE_EDGES[i], min(SCORE_EDGES[i + 1], 1.0)
    return f"{lo:g} to {hi:g}"


def sort_periods(tags) -> list[str]:
    return sorted(set(tags), key=lambda t: (Period.parse(t).month_index, Period.parse(t).n_months))


def months_of(quarter: str) -> list[str]:
    return [m.tag for m in Period.parse(quarter).months()]


def quarter_of(month: str) -> str:
    return Period.parse(month).quarter().tag


# ------------------------------------------------------------------ reading

class Analysis:
    """The prepared tables of one run."""

    def __init__(self, run: str | Path):
        self.run = Path(run)
        self.dir = self.run / "analysis"
        path = self.dir / "_build.json"
        self.log = json.loads(path.read_text()) if path.exists() else {}

    @property
    def ready(self) -> bool:
        return bool(self.log.get("built_utc"))

    def periods(self, frequency: str) -> list[str]:
        base = self.dir / frequency
        return sort_periods(p.name.removeprefix("period=") for p in base.glob("period=*")
                            if (p / "hits.parquet").exists())

    @lru_cache(maxsize=None)
    def static(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "static" / f"{name}.parquet")

    @lru_cache(maxsize=None)
    def _one(self, frequency: str, tag: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / frequency / f"period={tag}" / f"{name}.parquet")

    def table(self, frequency: str, name: str, periods: list[str] | None = None) -> pd.DataFrame:
        """One table of every period (or of `periods`), stacked, with a `period` column."""
        tags = periods if periods is not None else self.periods(frequency)
        parts = [self._one(frequency, t, name).assign(period=t) for t in tags]
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out["period"] = pd.Categorical(out["period"], categories=sort_periods(tags), ordered=True)
        return out

    def known_mines_text(self) -> str:
        k = self.log.get("known_mines", {})
        parts = []
        if k.get("amw_years"):
            parts.append(f"AMW's yearly detections of {_years(k['amw_years'])}")
        if k.get("our_years"):
            parts.append(f"our own quarterly detections of {_years(k['our_years'])}")
        return " and ".join(parts) + " (loose rule)" if parts else "none"


def _years(ys) -> str:
    ys = sorted(ys)
    if ys == list(range(ys[0], ys[-1] + 1)) and len(ys) > 2:
        return f"{ys[0]} to {ys[-1]}"
    return ", ".join(map(str, ys))


# ------------------------------------------------------------------ maps

def to_cells(period_tiles: pd.DataFrame, tiles: pd.DataFrame, deg: float | None = None,
             by: list[str] | None = None) -> pd.DataFrame:
    """Tile rows of one or more periods -> map cells (by default the builder's 0.25 degree cells).

    Means are weighted by the chips scored in each tile, so a cell's mean is the mean over
    its chips; counts are summed. `by`: further columns to keep apart (e.g. period).
    """
    t = period_tiles.merge(tiles[["tile", "lon", "lat", "cell_lon", "cell_lat", "subregion", "country"]], on="tile")
    if deg is not None:
        t["cell_lon"] = np.floor(t["lon"] / deg) * deg + deg / 2
        t["cell_lat"] = np.floor(t["lat"] / deg) * deg + deg / 2
    keys = ["cell_lon", "cell_lat", *(by or [])]
    w = t["scored"].astype(float)
    means = ["n_clear_mean", "share_n_clear_0", "share_n_clear_under_1", "empty_share_mean",
             "share_empty_over_half", "cs_mean_mean", "thin_share_mean", "confidence_mean", "vote_sd_mean"]
    for m in means:
        t[f"_{m}"] = t[m].astype(float) * w
    g = t.groupby(keys, observed=True)
    out = g.agg(chips=("chips", "sum"), scored=("scored", "sum"), raw=("raw", "sum"), loose=("loose", "sum"),
                stringent=("stringent", "sum"), andes=("andes", "sum"), near_raw=("near_raw", "sum"),
                confidence_max=("confidence_max", "max"), tiles=("tile", "size")).reset_index()
    sums = g[[f"_{m}" for m in means]].sum().reset_index(drop=True)
    for m in means:
        out[m] = sums[f"_{m}"].to_numpy() / out["scored"].replace(0, np.nan).to_numpy()
    for r in ("raw", "loose", "stringent", "andes", "near_raw"):
        out[f"{r}_per_1000"] = 1000 * out[r] / out["scored"].replace(0, np.nan)
    return out


# ------------------------------------------------------------------ detections through time

def detection_matrix(hits: pd.DataFrame, rule: str, periods: list[str] | None = None) -> pd.DataFrame:
    """chip x period: 1.0 detected, 0.0 scored and not detected, NaN not scored."""
    v = hits[rule].astype(float).where(hits["scored"])
    m = pd.DataFrame({"chip_id": hits["chip_id"], "period": hits["period"].astype(str), "v": v}) \
        .pivot(index="chip_id", columns="period", values="v")
    cols = sort_periods(periods if periods is not None else m.columns)
    return m.reindex(columns=cols)


def value_matrix(hits: pd.DataFrame, column: str, periods: list[str] | None = None) -> pd.DataFrame:
    m = pd.DataFrame({"chip_id": hits["chip_id"], "period": hits["period"].astype(str),
                      "v": hits[column].astype(float)}).pivot(index="chip_id", columns="period", values="v")
    return m.reindex(columns=sort_periods(periods if periods is not None else m.columns))


def pattern_strings(matrix: pd.DataFrame) -> pd.Series:
    """Each chip's periods as text: D detected, M missed (scored, not detected), · not scored."""
    a = matrix.to_numpy()
    chars = np.where(np.isnan(a), "·", np.where(a == 1, "D", "M"))
    return pd.Series(["".join(r) for r in chars], index=matrix.index, name="pattern")


def gap_table(matrix: pd.DataFrame, values: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Every run of misses that follows a detection, one row per missed period.

    A run between two detections is *closed* (the chip came back); a run after a chip's
    last detection is *open* (not detected since, up to the last period). Periods without
    a score (·) neither break nor extend a run: they are skipped.
    Columns: chip_id, gap (a number per run), period, position (1 = first missed period
    of the run), length (the run's missed periods), closed, start (last detected period
    before it), end (the period it came back, closed runs only), and, for each table in
    `values` (e.g. {"n_clear": value_matrix(hits, "n_clear")}), its value in that period
    and the chip's mean over its detected periods (`<name>_when_detected`).
    """
    cols = list(matrix.columns)
    a = matrix.to_numpy()
    rows = []
    gap_no = 0
    for r, chip in enumerate(matrix.index):
        seen = [(j, a[r, j]) for j in range(len(cols)) if not np.isnan(a[r, j])]
        detected_any = False
        run, start = [], None
        for j, v in seen:
            if v == 1:
                if detected_any and run:
                    gap_no += 1
                    rows += [(chip, gap_no, cols[k], i + 1, len(run), True, cols[start], cols[j]) for i, k in enumerate(run)]
                detected_any, run, start = True, [], j
            elif detected_any:
                run.append(j)
        if detected_any and run:
            gap_no += 1
            rows += [(chip, gap_no, cols[k], i + 1, len(run), False, cols[start], None) for i, k in enumerate(run)]
    out = pd.DataFrame(rows, columns=["chip_id", "gap", "period", "position", "length", "closed", "start", "end"])
    for name, vm in (values or {}).items():
        vm = vm.reindex(index=matrix.index, columns=cols)
        long = vm.stack(future_stack=True).rename(name).reset_index()
        long.columns = ["chip_id", "period", name]
        out = out.merge(long, on=["chip_id", "period"], how="left")
        when = (vm.where(matrix == 1)).mean(axis=1).rename(f"{name}_when_detected")
        out = out.merge(when, left_on="chip_id", right_index=True, how="left")
    return out


def pair_counts(matrix: pd.DataFrame, a: str, b: str) -> dict[str, int]:
    """How many chips were detected in period a and/or b (both scored)."""
    x, y = matrix[a], matrix[b]
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    return {"in both": int(((x == 1) & (y == 1)).sum()), f"only in {a}": int(((x == 1) & (y == 0)).sum()),
            f"only in {b}": int(((x == 0) & (y == 1)).sum()),
            "not scored in one of them": int(((matrix[a] == 1) | (matrix[b] == 1)).sum() - ((x == 1) | (y == 1)).sum())}


def retention(matrix: pd.DataFrame) -> pd.DataFrame:
    """For every ordered pair of periods (a, b): of the chips detected in a and scored in b,
    the share also detected in b; with the counts behind it. Long table, one row per pair."""
    cols = list(matrix.columns)
    a = matrix.to_numpy()
    rows = []
    for i, pa in enumerate(cols):
        det_a = a[:, i] == 1
        for j, pb in enumerate(cols):
            scored_b = ~np.isnan(a[:, j])
            base = det_a & scored_b
            kept = base & (a[:, j] == 1)
            rows.append({"from": pa, "to": pb, "detected in from": int(base.sum()),
                         "also in to": int(kept.sum()), "share": kept.sum() / base.sum() if base.sum() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ statistics

def wilson(k, n, z: float = 1.96):
    """95% Wilson interval of a share k / n (works for shares near 0 and 1). Returns (low, high)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def auc(worse: np.ndarray, better: np.ndarray) -> float:
    """P(a value from `worse` < a value from `better`), ties counted half (Mann-Whitney AUC).

    0.5 = the measure does not tell the two groups apart; 1 = it separates them perfectly.
    """
    worse, better = np.asarray(worse, float), np.asarray(better, float)
    worse, better = worse[np.isfinite(worse)], better[np.isfinite(better)]
    if not len(worse) or not len(better):
        return np.nan
    allv = np.concatenate([worse, better])
    ranks = pd.Series(allv).rank().to_numpy()
    r_better = ranks[len(worse):].sum()
    n1, n2 = len(better), len(worse)
    return (r_better - n1 * (n1 + 1) / 2) / (n1 * n2)


def within_chip_auc(long: pd.DataFrame, measure: str, direction: int, max_chips: int = 20_000) -> tuple[float, int, int]:
    """AUC computed only between periods of the same chip (missed against detected).

    `long`: rows of chip-periods with columns chip_id, detected (bool) and `measure`. For
    every chip with at least one detected and one missed period, every (missed, detected)
    pair of its periods is compared: does the missed period have the worse view? Pooled over
    chips. Because each chip is compared only with itself, where it lies (cloudy region,
    terrain) cannot explain the result; this is the idea of a chip fixed effect.
    Returns (AUC, chips used, pairs compared). At most `max_chips` chips (a fixed random set).
    """
    d = long[["chip_id", "detected", measure]].dropna()
    has = d.groupby("chip_id")["detected"].agg(["min", "max"])
    chips = has.index[(has["min"] == 0) & (has["max"] == 1)]
    if len(chips) > max_chips:
        chips = pd.Index(np.random.default_rng(0).choice(chips, max_chips, replace=False))
    d = d[d["chip_id"].isin(chips)]
    x = d[measure].to_numpy(float) * direction
    m, dd = d[~d["detected"].astype(bool)], d[d["detected"].astype(bool)]
    pairs = m[["chip_id"]].assign(xm=x[~d["detected"].astype(bool).to_numpy()]).merge(
        dd[["chip_id"]].assign(xd=x[d["detected"].astype(bool).to_numpy()]), on="chip_id")
    if not len(pairs):
        return np.nan, 0, 0
    score = (pairs["xm"] < pairs["xd"]).mean() + 0.5 * (pairs["xm"] == pairs["xd"]).mean()
    return float(score), int(len(chips)), int(len(pairs))


def quantile_bins(x: pd.Series, n: int = 10) -> pd.Series:
    """Bin a measure into (at most) n groups of about equal size; labels are the value ranges."""
    try:
        b = pd.qcut(x, n, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=x.index)
    return b


def rate_by_bins(df: pd.DataFrame, measure: str, outcome: str, n: int = 10) -> pd.DataFrame:
    """Share of rows with outcome == True per quantile group of `measure`, with a 95% interval."""
    d = df[[measure, outcome]].dropna()
    d["bin"] = quantile_bins(d[measure], n)
    g = d.groupby("bin", observed=True).agg(x=(measure, "median"), lo=(measure, "min"), hi=(measure, "max"),
                                           n=(outcome, "size"), k=(outcome, "sum")).reset_index(drop=True)
    g["share"] = g["k"] / g["n"]
    g["low"], g["high"] = wilson(g["k"], g["n"])
    return g


# ------------------------------------------------------------------ AMW's persistence rule (recipe A)

def yearly_status(amw: pd.DataFrame, rule: str = "det_loose", provisional_rule: str = "det_stringent") -> pd.DataFrame:
    """Every chip in AMW's yearly files under AMW's persistence rule (core/persistence.py, recipe A).

    A chip is *confirmed* with onset year Y if AMW's yearly picture detected it (loose rule)
    in Y and again in Y + 1. The last year AMW has can only be confirmed once the next year
    exists: a chip detected then (and not confirmed earlier) is *provisional* if the stricter
    single-look rule (stringent) also holds, else it is not in AMW's published record yet.
    Columns: chip_id, years (detected years as text), confirmed_onset, status.
    """
    y = amw[(amw["frequency"] == "yearly")]
    last = int(y["year"].max())
    det = y[y[rule]].groupby("chip_id")["year"].apply(lambda s: sorted(set(int(v) for v in s)))
    strict_last = set(y.loc[y[provisional_rule] & (y["year"] == last), "chip_id"])
    rows = []
    for chip, years in det.items():
        s = set(years)
        onset = next((yr for yr in years if yr + 1 in s), None)
        if onset is not None:
            status = "confirmed"
        elif last in s and chip in strict_last:
            status = f"provisional ({last} only)"
        elif last in s:
            status = f"{last} only, below the provisional rule"
        else:
            status = "never confirmed"
        rows.append({"chip_id": chip, "years": " ".join(map(str, years)), "confirmed_onset": onset, "status": status})
    return pd.DataFrame(rows, columns=["chip_id", "years", "confirmed_onset", "status"])


def quarterly_onset(matrix: pd.DataFrame, window: int = 4) -> pd.Series:
    """The first quarter a chip was detected in that is followed by another detection within
    `window` quarters (the quarterly counterpart of recipe A's one-year window). NaN if none."""
    cols = list(matrix.columns)
    a = matrix.to_numpy()
    out = np.full(len(a), None, dtype=object)
    for r in range(len(a)):
        det = np.flatnonzero(a[r] == 1)
        for i, j in enumerate(det):
            if i + 1 < len(det) and det[i + 1] - j <= window:
                out[r] = cols[j]
                break
    return pd.Series(out, index=matrix.index, name="quarterly_onset")


# ------------------------------------------------------------------ distributions from histograms

def hist_quantiles(h: pd.DataFrame, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[float, float]:
    """Quantiles of a variable from its histogram rows (left, right, chips), interpolated
    linearly within the bin (accurate to the bin width; the open last bin gives its left edge)."""
    h = h.sort_values("left")
    total = h["chips"].sum()
    if not total:
        return {q: np.nan for q in qs}
    cum = h["chips"].cumsum().to_numpy()
    left, right, n = h["left"].to_numpy(), h["right"].to_numpy(), h["chips"].to_numpy()
    out = {}
    for q in qs:
        i = int(np.searchsorted(cum, q * total))
        i = min(i, len(cum) - 1)
        before = cum[i] - n[i]
        frac = (q * total - before) / n[i] if n[i] else 0
        out[q] = left[i] if not np.isfinite(right[i]) else left[i] + frac * (right[i] - left[i])
    return out


def hist_share(h: pd.DataFrame, below: float) -> float:
    """Share of chips with a value below `below` (exact when `below` is a bin edge)."""
    total = h["chips"].sum()
    return h.loc[h["right"] <= below + 1e-12, "chips"].sum() / total if total else np.nan


def period_start(tag: str) -> pd.Timestamp:
    return pd.Timestamp(Period.parse(tag).start)


def period_middle(tag: str) -> pd.Timestamp:
    p = Period.parse(tag)
    return pd.Timestamp(p.start) + (pd.Timestamp(p.end) - pd.Timestamp(p.start)) / 2


def period_span(tag: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(first day, first day after) of a month, a quarter or a year ('2024', as in AMW's yearly files)."""
    tag = str(tag)
    if tag.isdigit() and len(tag) == 4:
        return pd.Timestamp(f"{tag}-01-01"), pd.Timestamp(f"{int(tag) + 1}-01-01")
    p = Period.parse(tag)
    return pd.Timestamp(p.start), pd.Timestamp(p.end)
