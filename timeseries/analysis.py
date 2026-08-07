#!/usr/bin/env python3
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
"""
analysis.py — statistical comparison of MONTHLY vs QUARTERLY composites for the
mining time series (phase-one pilot, one AOI, one year).

RESEARCH QUESTION
-----------------
Can monthly Sentinel-2 median composites support a mining time series over the
Amazon, or does wet-season cloud cover make monthly detections too noisy /
too gap-ridden, so that quarterly composites (or a hybrid) are the better
econometric base frequency?

DESIGN (what the overnight run produced; see panel.py)
------------------------------------------------------
A balanced chip panel: ~N chips x 16 periods (12 monthly + 4 quarterly windows
of the same calendar year, same AOI, same model). Per chip-period:
  prob        continuous CNN score in [0,1] (NOT thresholded at source)
  clear_frac  share of the chip's pixels with >=1 clear scene (cloud flag)
  n_clear_*   per-pixel count of scenes entering the median (precision proxy)

KEY METHODOLOGICAL CAUTIONS (read before citing any number)
-----------------------------------------------------------
1. Model output is NOT ground truth. Every "detection" is a CNN score; the
   monthly-vs-quarterly comparison is therefore about the STABILITY and
   INTERNAL CONSISTENCY of the measurement instrument, not about accuracy.
   Validation against imagery / published AMW outputs is a separate step.
2. Missingness is almost certainly MNAR: clouds follow the wet season, and
   mining activity (river levels, access) is itself seasonal. Estimates that
   drop cloudy chips are NOT unbiased for the seasonal signal — we quantify
   the missingness so the interpolation strategy can model it, not ignore it.
3. n = 12 months (11 first differences) — every time-series statistic below is
   DESCRIPTIVE. No asymptotics, no causal claims. We use rank/exact tests where
   possible and report effect sizes, not stars.
4. The quarterly composite is not the mean of its months: the median over ~18
   scenes differs non-linearly from medians over ~6. Monthly-vs-quarterly
   disagreement conflates temporal signal, composite non-linearity, and noise;
   the within-quarter analyses (§F) try to separate these under the stated
   slow-dynamics assumption.

Usage:
    python timeseries/analysis.py                      # default dirs
    python timeseries/analysis.py --panels_dir ... --out_dir ...

Outputs: <out_dir>/REPORT.md, figs/*.png, tables/*.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless: runs unattended at the end of the overnight job
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("analysis")

# ----------------------------------------------------------------------------#
# constants / definitions                                                     #
# ----------------------------------------------------------------------------#
UTM_CRS = "EPSG:32719"        # UTM 19S — correct metric CRS for Madre de Dios
TAU_OBSERVED = 0.95           # chip counts as OBSERVED if clear_frac >= tau
TAU_SENSITIVITY = 0.50        # laxer definition, reported alongside
T_MAIN = 0.4                  # working threshold (model config t0.4)
T_ALT = 0.5                   # engine default, for sensitivity
T_GRID = np.round(np.arange(0.30, 0.61, 0.05), 2)
NEAR_BAND = 0.05              # "near-threshold" = |prob - T_MAIN| <= NEAR_BAND

MONTH_RE = re.compile(r"^m(\d{4})-(\d{2})$")
QUARTER_RE = re.compile(r"^q(\d{4})Q(\d)$")


def quarter_of_month(label: str) -> str | None:
    """'m2025-04' -> 'q2025Q2'; None for non-month labels."""
    m = MONTH_RE.match(label)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return f"q{y}Q{(mo - 1) // 3 + 1}"


# ----------------------------------------------------------------------------#
# loading                                                                     #
# ----------------------------------------------------------------------------#
def load_panels(panels_dir: Path) -> gpd.GeoDataFrame:
    files = sorted(panels_dir.glob("panel_*.parquet"))
    if not files:
        sys.exit(f"No panel files in {panels_dir} — run run_trials.py first.")
    parts = [gpd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    log.info("Loaded %d chip-period rows from %d panels", len(df), len(files))
    _check_provenance(df, files)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=parts[0].crs)


def _check_provenance(df: pd.DataFrame, files: list[Path]) -> None:
    """Refuse to silently pool panels produced under different configurations.

    This glob has no year, model or threshold filter, and ``pd.concat`` fills missing
    columns with NaN without complaint — so panels from two different models, two cloud
    thresholds, or schema v1 and v2 would merge into one dataframe and produce a
    plausible-looking but meaningless report. Panel schema v2 (panel.py) records the
    configuration in every row precisely so this check is possible.

    Mixed ``clear_threshold`` is fatal: it changes the composite the CNN saw, so the
    probabilities are not comparable. Everything else warns loudly.
    """
    if "schema_version" not in df.columns:
        log.warning(
            "Panels carry no provenance columns (schema v1, written before "
            "2026-08-04). Cannot verify that they share a model and cloud threshold — "
            "results are only as trustworthy as your bookkeeping.")
        return

    if df["schema_version"].nunique(dropna=False) > 1:
        log.warning("MIXED PANEL SCHEMAS: %s. Rows from older panels will have NaN in "
                    "the newer columns.", sorted(df["schema_version"].dropna().unique()))

    if "clear_threshold" in df.columns and df["clear_threshold"].nunique(dropna=True) > 1:
        sys.exit(
            f"FATAL: panels in this directory were built with different "
            f"clear_threshold values {sorted(df['clear_threshold'].dropna().unique())}. "
            f"The cloud mask is applied before the median composite (gee.py:132), so the "
            f"probabilities are not comparable. Separate them into different --panels_dir.")

    for col, label in (("model", "model file"),
                       ("model_sha256", "model weights"),
                       ("chip_size", "chip size"),
                       ("stride", "stride"),
                       ("git_commit", "code version")):
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            log.warning("MIXED %s across panels: %s — check this is intended.",
                        label, sorted(map(str, df[col].dropna().unique()))[:5])

    n_periods = df["period"].nunique()
    log.info("Provenance OK: schema v%s, model=%s, clear_threshold=%s, %d periods "
             "from %d files",
             df["schema_version"].iloc[0], df["model"].iloc[0],
             df["clear_threshold"].iloc[0], n_periods, len(files))


def load_tile_stats(tiles_dir: Path) -> pd.DataFrame:
    files = sorted(tiles_dir.glob("tile_stats_*.csv"))
    return (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            if files else pd.DataFrame())


# ----------------------------------------------------------------------------#
# helpers                                                                     #
# ----------------------------------------------------------------------------#
def dissolved_area_km2(g: gpd.GeoDataFrame) -> float:
    """Area of the UNION of (overlapping, stride-24) chip squares, in km².
    Union first — summing overlapping chips would double-count by ~4x."""
    if len(g) == 0:
        return 0.0
    return float(g.to_crs(UTM_CRS).union_all().area) / 1e6


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


class Report:
    """Accumulates markdown; every analysis section appends here so a partial
    run (e.g. smoke test with 2 periods) still yields a readable report."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, *ls: str):
        self.lines.extend(ls)

    def table(self, df: pd.DataFrame, float_nd=3):
        # plain-text block instead of markdown table: avoids the `tabulate`
        # dependency and keeps alignment for wide tables
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            self.add("```", df.round(float_nd).to_string(), "```", "")

    def write(self, path: Path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        log.info("Report -> %s", path)


# ----------------------------------------------------------------------------#
# A. coverage / missingness                                                   #
# ----------------------------------------------------------------------------#
def section_coverage(df: pd.DataFrame, rep: Report, figs: Path, tables: Path):
    rep.add("## A. Cloud coverage & missingness (the case for the cloud flag)", "")
    per = (df.groupby(["cadence", "period"])
             .agg(n_chips=("chip_id", "size"),
                  mean_clear_frac=("clear_frac", "mean"),
                  share_fully_obs=("clear_frac", lambda s: np.mean(s >= TAU_OBSERVED)),
                  share_half_obs=("clear_frac", lambda s: np.mean(s >= TAU_SENSITIVITY)),
                  share_fully_missing=("clear_frac", lambda s: np.mean(s == 0)),
                  mean_n_clear=("n_clear_mean", "mean"))
             .reset_index())
    per.to_csv(tables / "coverage_by_period.csv", index=False)
    rep.add(f"Chip counts as *observed* if `clear_frac >= {TAU_OBSERVED}` "
            f"(sensitivity: {TAU_SENSITIVITY}).", "")
    rep.table(per)

    mo = per[per.cadence == "monthly"].sort_values("period")
    qu = per[per.cadence == "quarterly"].sort_values("period")
    if len(mo):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mo.period, mo.share_fully_obs, "o-", label=f"monthly, share chips ≥{TAU_OBSERVED} clear")
        ax.plot(mo.period, mo.mean_clear_frac, "s--", label="monthly, mean clear_frac")
        if len(qu):
            # place quarterly points at their middle month for visual comparison
            qx = {f"q{y}Q{q}": f"m{y}-{(int(q) - 1) * 3 + 2:02d}"
                  for y, q in (QUARTER_RE.match(p).groups() for p in qu.period)}
            ax.plot([qx[p] for p in qu.period], qu.share_fully_obs, "^",
                    markersize=10, label=f"quarterly, share ≥{TAU_OBSERVED}")
        ax.set_ylim(0, 1.02); ax.set_ylabel("share / fraction")
        ax.set_title("Observability by period (wet season ≈ Nov–Mar)")
        ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout()
        fig.savefig(figs / "fig_A_coverage_by_period.png", dpi=150); plt.close(fig)
        rep.add("![coverage](figs/fig_A_coverage_by_period.png)", "")

    # chronic gaps: chips rarely observed at monthly cadence are exactly the
    # locations where a monthly series would need heavy interpolation
    mdf = df[df.cadence == "monthly"]
    if len(mdf):
        months_obs = (mdf.assign(obs=mdf.clear_frac >= TAU_OBSERVED)
                        .groupby("chip_id")["obs"].sum())
        counts = months_obs.value_counts().sort_index()
        counts.to_csv(tables / "months_observed_histogram.csv")
        n_months = mdf.period.nunique()
        rep.add(f"Distribution of #months (of {n_months}) each chip is fully "
                "observed — chips at the low end are *chronic* gaps "
                "(interpolation there is extrapolation):", "")
        rep.table(counts.rename("n_chips").to_frame())
    return per


# ----------------------------------------------------------------------------#
# B. fill-value verification                                                  #
# ----------------------------------------------------------------------------#
def section_fill_check(tile_stats: pd.DataFrame, rep: Report):
    rep.add("## B. Empirical verification: cloud gap == all-zero pixels", "")
    if tile_stats.empty or "fill0_mismatch" not in tile_stats:
        rep.add("_No tile stats found — section skipped._", "")
        return
    mm = tile_stats["fill0_mismatch"].dropna()
    rep.add(f"Per-tile disagreement between `N_CLEAR==0` (coverage raster) and "
            f"`all 13 bands == 0` (imagery): mean {fmt(mm.mean(), 6)}, "
            f"max {fmt(mm.max(), 6)} over {len(mm)} tile-periods.",
            "",
            "Interpretation: values ≈ 0 CONFIRM the pipeline card's assumption "
            "that Earth Engine fills fully-masked pixels with 0, and that the "
            "all-zero heuristic is a valid retroactive cloud flag for already-"
            "cached imagery. Non-negligible values would mean gaps are only "
            "reliably identifiable via the coverage raster (keep using it "
            "either way — it also gives N_CLEAR as a precision measure).", "")


# ----------------------------------------------------------------------------#
# C. detection series & cloud bias                                            #
# ----------------------------------------------------------------------------#
def section_series(df: gpd.GeoDataFrame, rep: Report, figs: Path, tables: Path):
    rep.add("## C. Detected mining area: naive vs cloud-aware", "")
    rows = []
    for (cad, period), g in df.groupby(["cadence", "period"]):
        obs = g[g.clear_frac >= TAU_OBSERVED]
        for t in (T_MAIN, T_ALT):
            rows.append({
                "cadence": cad, "period": period, "threshold": t,
                "area_naive_km2": dissolved_area_km2(g[g.prob > t]),
                "area_observed_km2": dissolved_area_km2(obs[obs.prob > t]),
                "n_pos_naive": int((g.prob > t).sum()),
                "n_pos_observed": int((obs.prob > t).sum()),
                "share_chips_observed": float(np.mean(g.clear_frac >= TAU_OBSERVED)),
            })
    ser = pd.DataFrame(rows)
    ser.to_csv(tables / "area_series.csv", index=False)
    rep.add(f"`area_naive` uses ALL chips (as the raw pipeline would); "
            f"`area_observed` restricts to chips with clear_frac ≥ {TAU_OBSERVED}. "
            "The DIFFERENCE is the mechanical cloud artefact a naive series "
            "would carry. Areas are km² of the dissolved union of positive "
            "480 m chips (union, not sum — chips overlap by construction).", "")
    rep.table(ser[ser.threshold == T_MAIN].drop(columns="threshold"))

    m = ser[(ser.cadence == "monthly") & (ser.threshold == T_MAIN)].sort_values("period")
    q = ser[(ser.cadence == "quarterly") & (ser.threshold == T_MAIN)].sort_values("period")
    if len(m):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(m.period, m.area_naive_km2, "o--", alpha=.6, label="monthly, naive")
        ax.plot(m.period, m.area_observed_km2, "o-", label="monthly, cloud-aware")
        if len(q):
            qx = {f"q{y}Q{qq}": f"m{y}-{(int(qq) - 1) * 3 + 2:02d}"
                  for y, qq in (QUARTER_RE.match(p).groups() for p in q.period)}
            ax.plot([qx[p] for p in q.period], q.area_observed_km2, "^",
                    markersize=10, label="quarterly, cloud-aware")
        ax.set_ylabel(f"km² (t={T_MAIN})"); ax.tick_params(axis="x", rotation=45)
        ax.set_title("Detected mining area"); ax.legend(); fig.tight_layout()
        fig.savefig(figs / "fig_C_area_series.png", dpi=150); plt.close(fig)
        rep.add("![area](figs/fig_C_area_series.png)", "")

    # stability summaries (descriptive; n is tiny)
    for cad, sub in ser[ser.threshold == T_MAIN].groupby("cadence"):
        a = sub.sort_values("period").area_observed_km2.to_numpy()
        if len(a) >= 3:
            cv = a.std(ddof=1) / a.mean() if a.mean() > 0 else np.nan
            d = np.diff(a)
            rep.add(f"- **{cad}** cloud-aware series: mean {fmt(a.mean(),1)} km², "
                    f"CV {fmt(cv)}, mean |Δ| between consecutive periods "
                    f"{fmt(np.abs(d).mean(),1)} km², "
                    f"lag-1 autocorr {fmt(pd.Series(a).autocorr(),2)} "
                    f"(n={len(a)} — descriptive only).")
    rep.add("")
    return ser


# ----------------------------------------------------------------------------#
# D. is the 'signal' just clouds? confounding checks                          #
# ----------------------------------------------------------------------------#
def section_confounding(df: pd.DataFrame, ser: pd.DataFrame, rep: Report,
                        figs: Path):
    rep.add("## D. Confounding: does apparent change track cloud cover?", "")
    m = ser[(ser.cadence == "monthly") & (ser.threshold == T_MAIN)].sort_values("period")
    if len(m) >= 6:
        d_area = np.diff(m.area_observed_km2.to_numpy())
        d_cov = np.diff(m.share_chips_observed.to_numpy())
        rho, p = stats.spearmanr(d_area, d_cov)
        slope, icpt, r, p_ols, se = stats.linregress(d_cov, d_area)
        rep.add(f"First differences, monthly, cloud-aware series (n={len(d_area)} pairs):",
                f"- Spearman ρ(Δarea, Δcoverage) = {fmt(rho,2)} (p={fmt(p,2)}, exact "
                "p-values unreliable at this n; read the sign and size).",
                f"- OLS: Δarea = {fmt(icpt,1)} + {fmt(slope,1)}·Δcoverage, R²={fmt(r**2,2)}.",
                "",
                "If ρ is clearly positive, month-to-month movements in the "
                "*cloud-aware* series still co-move with observability — i.e. "
                "compositional missingness (WHICH chips are observable changes "
                "month to month) leaks into the series. The fix in the "
                "econometric stage is a balanced-panel estimator (only chips "
                "observed in both adjacent months contribute to Δ) or "
                "explicit imputation — not deletion.", "")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(d_cov, d_area)
        ax.set_xlabel("Δ share of chips observed"); ax.set_ylabel("Δ area km²")
        ax.set_title("Monthly first differences"); fig.tight_layout()
        fig.savefig(figs / "fig_D_area_vs_coverage.png", dpi=150); plt.close(fig)
        rep.add("![confound](figs/fig_D_area_vs_coverage.png)", "")
    else:
        rep.add("_Fewer than 6 monthly periods — differencing check skipped._", "")

    # chip-level: are classification flips associated with coverage changes?
    mdf = df[df.cadence == "monthly"]
    if mdf.period.nunique() >= 2:
        wide_p = mdf.pivot_table(index="chip_id", columns="period", values="prob")
        wide_c = mdf.pivot_table(index="chip_id", columns="period", values="clear_frac")
        cols = sorted(wide_p.columns)
        flips, dcov_flip, dcov_stable = 0, [], []
        for a, b in zip(cols[:-1], cols[1:]):
            ok = (wide_c[a] >= TAU_OBSERVED) & (wide_c[b] >= TAU_OBSERVED)
            pa, pb = wide_p.loc[ok, a], wide_p.loc[ok, b]
            flip = (pa > T_MAIN) != (pb > T_MAIN)
            dc = (wide_c.loc[ok, b] - wide_c.loc[ok, a]).abs()
            flips += int(flip.sum())
            dcov_flip += dc[flip].tolist()
            dcov_stable += dc[~flip].tolist()
        if dcov_flip and dcov_stable:
            u = stats.mannwhitneyu(dcov_flip, dcov_stable, alternative="greater")
            rep.add(f"Chip-level: {flips} month-to-month classification flips at "
                    f"t={T_MAIN} among consecutively-observed chips. "
                    f"|Δclear_frac| for flippers vs stable chips: medians "
                    f"{fmt(np.median(dcov_flip))} vs {fmt(np.median(dcov_stable))} "
                    f"(Mann-Whitney one-sided p={fmt(u.pvalue,3)}). A clear gap "
                    "would implicate residual partial-cloud noise (chips passing "
                    f"the τ={TAU_OBSERVED} filter but still degraded).", "")


# ----------------------------------------------------------------------------#
# E. monthly vs quarterly consistency                                          #
# ----------------------------------------------------------------------------#
def section_consistency(df: gpd.GeoDataFrame, rep: Report, figs: Path,
                        tables: Path):
    rep.add("## E. Monthly ↔ quarterly consistency", "",
            "Comparisons are restricted to chips fully observed in the quarter "
            "AND in all three constituent months (balanced subsample) — "
            "disagreement here cannot be blamed on missingness, only on "
            "composite non-linearity, temporal aggregation, and noise.", "")
    qdf = df[df.cadence == "quarterly"]
    mdf = df[df.cadence == "monthly"]
    if qdf.empty or mdf.empty:
        rep.add("_Need both cadences — skipped._", "")
        return

    mdf = mdf.assign(q=mdf.period.map(quarter_of_month))
    rows = []
    fig, axes = plt.subplots(1, max(qdf.period.nunique(), 1),
                             figsize=(4 * qdf.period.nunique(), 4), squeeze=False)
    for ax, (qlab, qg) in zip(axes[0], qdf.groupby("period")):
        mg = mdf[mdf.q == qlab]
        if mg.empty:
            continue
        # balanced subsample
        p_m = mg.pivot_table(index="chip_id", columns="period", values="prob")
        c_m = mg.pivot_table(index="chip_id", columns="period", values="clear_frac")
        q_idx = qg.set_index("chip_id")
        common = p_m.index.intersection(q_idx.index)
        ok = (c_m.loc[common].ge(TAU_OBSERVED).all(axis=1)
              & (q_idx.loc[common, "clear_frac"] >= TAU_OBSERVED))
        ids = common[ok]
        if len(ids) < 50:
            rep.add(f"- **{qlab}**: only {len(ids)} balanced chips — skipped.")
            continue
        pq = q_idx.loc[ids, "prob"]
        pm_mean = p_m.loc[ids].mean(axis=1)
        pm_max = p_m.loc[ids].max(axis=1)

        rho = stats.spearmanr(pq, pm_mean).statistic
        mae = float((pq - pm_mean).abs().mean())
        # classification agreement at T_MAIN: quarterly vs monthly-mean and monthly-any
        yq = pq > T_MAIN
        ym_mean = pm_mean > T_MAIN
        ym_any = pm_max > T_MAIN
        kappa_mean = cohens_kappa(yq, ym_mean)
        kappa_any = cohens_kappa(yq, ym_any)

        # footprint IoU (dissolved geometries, quarterly vs monthly variants)
        qpos = qg[qg.chip_id.isin(ids) & (qg.prob > T_MAIN)]
        month_pos_any = mg[mg.chip_id.isin(ids) & (mg.prob > T_MAIN)]
        iou_any = geo_iou(qpos, month_pos_any)

        rows.append({"quarter": qlab, "n_balanced_chips": len(ids),
                     "spearman_q_vs_meanM": rho, "MAE_prob": mae,
                     "kappa_q_vs_meanM": kappa_mean, "kappa_q_vs_anyM": kappa_any,
                     "IoU_footprint_q_vs_anyM": iou_any,
                     "pos_rate_q": float(yq.mean()),
                     "pos_rate_meanM": float(ym_mean.mean()),
                     "pos_rate_anyM": float(ym_any.mean())})

        ax.scatter(pm_mean, pq, s=4, alpha=.3)
        ax.axhline(T_MAIN, color="r", lw=.5); ax.axvline(T_MAIN, color="r", lw=.5)
        ax.set_xlabel("mean of 3 monthly probs"); ax.set_ylabel("quarterly prob")
        ax.set_title(qlab)
    fig.tight_layout()
    fig.savefig(figs / "fig_E_mq_scatter.png", dpi=150); plt.close(fig)

    if rows:
        tab = pd.DataFrame(rows)
        tab.to_csv(tables / "mq_consistency.csv", index=False)
        rep.table(tab)
        rep.add("![mq](figs/fig_E_mq_scatter.png)", "",
                "Reading guide: high ρ/κ with `pos_rate_anyM > pos_rate_q` is "
                "the expected pattern if monthly composites are noisier — "
                "'any month positive' accumulates false positives 3 chances "
                "instead of 1. If instead `pos_rate_q` exceeds `anyM`, the "
                "quarterly composite is genuinely more sensitive (median over "
                "more clear scenes → cleaner signal), which argues for "
                "quarterly anchors in the econometric model.", "")


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's κ for two boolean vectors (no sklearn dependency)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def geo_iou(g1: gpd.GeoDataFrame, g2: gpd.GeoDataFrame) -> float:
    """IoU of two dissolved chip-footprints in metric CRS."""
    if len(g1) == 0 and len(g2) == 0:
        return np.nan
    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    u1 = g1.to_crs(UTM_CRS).union_all()
    u2 = g2.to_crs(UTM_CRS).union_all()
    inter = u1.intersection(u2).area
    union = u1.union(u2).area
    return float(inter / union) if union > 0 else np.nan


# ----------------------------------------------------------------------------#
# F. within-quarter noise bound                                               #
# ----------------------------------------------------------------------------#
def section_noise(df: pd.DataFrame, rep: Report, tables: Path):
    rep.add("## F. Within-quarter dispersion: an upper bound on monthly noise", "",
            "IDENTIFYING ASSUMPTION (state it, defend it, doubt it): true mining "
            "extent at a 480 m chip changes little WITHIN a quarter, so the "
            "dispersion of a chip's three monthly probabilities mostly reflects "
            "measurement noise (composite quality, atmospheric residuals). This "
            "is defensible for established pits, weakest during rapid frontier "
            "expansion — where within-quarter change is real. The bound is "
            "therefore an UPPER bound on noise.", "")
    mdf = df[df.cadence == "monthly"].assign(
        q=lambda d: d.period.map(quarter_of_month))
    qdf = df[df.cadence == "quarterly"].set_index(["chip_id", "period"])
    rows = []
    for qlab, mg in mdf.groupby("q"):
        p = mg.pivot_table(index="chip_id", columns="period", values="prob")
        c = mg.pivot_table(index="chip_id", columns="period", values="clear_frac")
        if p.shape[1] < 3:
            continue
        ok = c.ge(TAU_OBSERVED).all(axis=1)
        p = p.loc[ok]
        if p.empty:
            continue
        sd = p.std(axis=1, ddof=1)
        flip = ((p.gt(T_MAIN).sum(axis=1) % 3) != 0)  # not 0/3 or 3/3 positive
        # stratify by the quarterly score where available (confident vs ambiguous)
        try:
            pq = qdf.loc[[(i, qlab) for i in p.index], "prob"].to_numpy()
        except KeyError:
            pq = np.full(len(p), np.nan)
        conf_neg = pq < 0.1
        conf_pos = pq > 0.9
        rows.append({
            "quarter": qlab, "n_chips_balanced": len(p),
            "median_sd_prob": float(sd.median()),
            "p90_sd_prob": float(sd.quantile(.9)),
            "flip_rate_within_q": float(flip.mean()),
            "median_sd_conf_neg": float(sd[conf_neg].median()) if conf_neg.any() else np.nan,
            "median_sd_conf_pos": float(sd[conf_pos].median()) if conf_pos.any() else np.nan,
        })
    if rows:
        tab = pd.DataFrame(rows)
        tab.to_csv(tables / "within_quarter_noise.csv", index=False)
        rep.table(tab)
        rep.add("`flip_rate_within_q` = share of balanced chips whose "
                f"classification at t={T_MAIN} is not constant across the "
                "quarter's three months. `median_sd_conf_neg/pos` isolate chips "
                "the quarterly composite scores confidently (<0.1 / >0.9): "
                "dispersion THERE is almost surely pure noise. If the "
                "confident-chip sd is small but the overall flip rate is high, "
                "monthly noise is concentrated near the decision boundary — an "
                "argument for using the continuous prob (not the 0/1 flag) in "
                "the econometric model.", "")
    else:
        rep.add("_Not enough balanced monthly data — skipped._", "")


# ----------------------------------------------------------------------------#
# G. threshold sensitivity                                                    #
# ----------------------------------------------------------------------------#
def section_threshold(df: gpd.GeoDataFrame, rep: Report, figs: Path,
                      tables: Path):
    rep.add("## G. Threshold sensitivity", "")
    rows = []
    for (cad, period), g in df.groupby(["cadence", "period"]):
        obs = g[g.clear_frac >= TAU_OBSERVED]
        near = float(np.mean(np.abs(obs.prob - T_MAIN) <= NEAR_BAND)) if len(obs) else np.nan
        r = {"cadence": cad, "period": period, "share_near_threshold": near}
        for t in T_GRID:
            r[f"area_t{t}"] = dissolved_area_km2(obs[obs.prob > t])
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(tables / "threshold_sensitivity.csv", index=False)

    mean_near = tab.groupby("cadence")["share_near_threshold"].mean()
    rep.add("Mean share of observed chips within ±"
            f"{NEAR_BAND} of t={T_MAIN} (chips whose classification is "
            "fragile to the threshold choice):", "")
    rep.table(mean_near.to_frame())
    rep.add("If the monthly share exceeds the quarterly share, monthly "
            "composites push more probability mass toward the decision "
            "boundary — the fingerprint of noisier inputs.", "")

    fig, ax = plt.subplots(figsize=(6, 4))
    for cad, sub in tab.groupby("cadence"):
        areas = [sub[f"area_t{t}"].mean() for t in T_GRID]
        ax.plot(T_GRID, areas, "o-", label=f"{cad} (mean over periods)")
    ax.set_xlabel("threshold t"); ax.set_ylabel("mean detected area km²")
    ax.set_title("Area vs threshold"); ax.legend(); fig.tight_layout()
    fig.savefig(figs / "fig_G_threshold.png", dpi=150); plt.close(fig)
    rep.add("![thresh](figs/fig_G_threshold.png)", "")


# ----------------------------------------------------------------------------#
# H. precision vs number of clear scenes                                      #
# ----------------------------------------------------------------------------#
def section_nclear(df: pd.DataFrame, rep: Report, tables: Path):
    rep.add("## H. Does composite precision rise with clear-scene count?", "",
            "Mechanism: the median over k clear scenes has sampling variance "
            "falling in k. If monthly noise is driven by few-scene composites, "
            "chip-probability dispersion should DECLINE in `n_clear_mean`. We "
            "test this on quarterly-confident NEGATIVES (quarterly prob < 0.1) "
            "— for these, monthly variation is interpretable as noise.", "")
    mdf = df[df.cadence == "monthly"].assign(
        q=lambda d: d.period.map(quarter_of_month))
    qneg = (df[(df.cadence == "quarterly") & (df.prob < 0.1)]
            [["chip_id", "period"]].rename(columns={"period": "q"}))
    sub = mdf.merge(qneg, on=["chip_id", "q"], how="inner")
    sub = sub[sub.clear_frac >= TAU_OBSERVED]
    if sub.empty:
        rep.add("_No quarterly-confident negatives with observed months — skipped._", "")
        return
    bins = [1, 2, 3, 5, 8, np.inf]
    labels = ["1-2", "2-3", "3-5", "5-8", "8+"]
    sub = sub.assign(nclear_bin=pd.cut(sub.n_clear_mean, bins=bins,
                                       labels=labels, right=False))
    tab = (sub.groupby("nclear_bin", observed=True)
              .agg(n=("prob", "size"),
                   p95_prob=("prob", lambda s: float(np.quantile(s, .95))),
                   mean_prob=("prob", "mean"),
                   share_above_t=("prob", lambda s: float(np.mean(s > T_MAIN))))
              .reset_index())
    tab.to_csv(tables / "noise_vs_nclear.csv", index=False)
    rep.table(tab)
    rep.add("`p95_prob` and `share_above_t` on known-negative chips are "
            "empirical FALSE-POSITIVE tendencies by scene count. A monotone "
            "decline justifies using `n_clear` as a precision weight (or "
            "observation-error variance) in the state-space/interpolation "
            "model rather than a binary include/exclude rule.", "")


# ----------------------------------------------------------------------------#
# implications + limitations (written text, filled by the numbers above)      #
# ----------------------------------------------------------------------------#
def section_implications(rep: Report, df: pd.DataFrame | None = None):
    # The limitation text below quotes the number of monthly periods. Derive it from the
    # data rather than hardcoding 12, so a multi-year run does not silently ship a false
    # claim in its own limitations section.
    n_months = 12
    n_years = 1
    if df is not None and "period" in df:
        months = df.loc[df.cadence == "monthly", "period"].unique()
        n_months = len(months)
        n_years = len({m[1:5] for m in months if MONTH_RE.match(m)}) or 1
    scope = ("one year" if n_years == 1
             else f"{n_years} years")
    rep.add(
        "## I. Implications for the econometric interpolation strategy", "",
        "1. **Never treat a low score under low `clear_frac` as a zero.** The "
        "detection output cannot distinguish 'no mine' from 'no data'; the "
        "cloud flag now makes the distinction explicit. Any monthly series "
        "must be built from (prob, clear_frac, n_clear) triples, not from "
        "thresholded detections alone.",
        "2. **Missingness is seasonal, hence plausibly MNAR.** Section A "
        "documents the wet-season observability trough. Dropping cloudy "
        "months and interpolating linearly would systematically smooth over "
        "exactly the season where mining dynamics (river levels, access) "
        "differ. Prefer a model that treats clear_frac as an observation "
        "process: state-space / Kalman with time-varying observation noise "
        "(variance ∝ 1/n_clear, gated on clear_frac), or monthly estimates "
        "anchored to quarterly composites (section E tells you how much the "
        "two instruments disagree even without missingness).",
        "3. **Use the continuous probability, not the binary flag, as the "
        "measurement.** Sections F–G locate monthly noise near the decision "
        "boundary; thresholding early converts small score noise into large "
        "area noise. Threshold once, at the end, on the smoothed series.",
        "4. **Balanced-panel differencing.** For growth rates, difference only "
        "chips observed in both periods (section D shows why); report the "
        "share of area carried by the balanced subsample alongside.",
        "",
        "## J. Limitations (to carry into any write-up)", "",
        f"- One AOI (La Pampa corridor), {scope}, one model: no external "
        "validity claimed. The AOI was chosen for dense mining signal, i.e. "
        "non-randomly.",
        "- CNN scores are the measurement instrument, not ground truth; all "
        "consistency results are internal. Validation against published AMW "
        "quarterly outputs and/or manual photo-interpretation is the natural "
        "next step.",
        f"- n = {n_months} months ({max(n_months - 1, 0)} first differences): all "
        "time-series statistics are descriptive; no significance claims should "
        "survive into the paper without more periods or more AOIs.",
        "- Quarterly composites differ from monthly ones non-linearly (median "
        "over ~3× scenes); 'quarterly as truth' is an assumption of "
        "convenience in sections F/H, not a fact.",
        "- Sentinel-2 revisit (~5 days) aliases sub-monthly dynamics; monthly "
        "'signal' partly reflects WHICH days were clear (acquisition-timing "
        "noise), not only surface change.", "")


# ----------------------------------------------------------------------------#
# main                                                                        #
# ----------------------------------------------------------------------------#
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_dir", type=Path,
                    default=REPO_ROOT / "data/timeseries/panels")
    ap.add_argument("--tiles_dir", type=Path,
                    default=REPO_ROOT / "data/timeseries/tiles")
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "data/timeseries/analysis")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    figs = args.out_dir / "figs"; figs.mkdir(parents=True, exist_ok=True)
    tables = args.out_dir / "tables"; tables.mkdir(parents=True, exist_ok=True)

    df = load_panels(args.panels_dir)
    tile_stats = load_tile_stats(args.tiles_dir)

    rep = Report()
    rep.add("# Monthly vs quarterly composites — pilot analysis",
            f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}; "
            f"{df.period.nunique()} periods, {df.chip_id.nunique()} chips, "
            f"{len(df)} chip-period rows._",
            "",
            "Definitions: chip = 480 m × 480 m window (stride 240 m, overlapping); "
            f"observed = clear_frac ≥ {TAU_OBSERVED}; working threshold t = {T_MAIN} "
            "(model config); areas = dissolved unions in EPSG:32719. "
            "Read the docstring of analysis.py for the methodological cautions "
            "that govern interpretation of every section below.", "")

    # Each section is independent: a failure (e.g. smoke-test data with one
    # period) degrades to a note in the report instead of killing the run.
    sections = [
        ("A", lambda: section_coverage(df, rep, figs, tables)),
        ("B", lambda: section_fill_check(tile_stats, rep)),
        ("C", lambda: setattr(main, "_ser", section_series(df, rep, figs, tables))),
        ("D", lambda: section_confounding(df, getattr(main, "_ser", pd.DataFrame()),
                                          rep, figs)),
        ("E", lambda: section_consistency(df, rep, figs, tables)),
        ("F", lambda: section_noise(df, rep, tables)),
        ("G", lambda: section_threshold(df, rep, figs, tables)),
        ("H", lambda: section_nclear(df, rep, tables)),
        ("I/J", lambda: section_implications(rep, df)),
    ]
    for name, fn in sections:
        try:
            fn()
        except Exception as e:
            log.exception("Section %s failed", name)
            rep.add(f"_Section {name} failed: {e!r} — see log._", "")

    rep.write(args.out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
