import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="3 Distance and confidence")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    import analysis_tables as at
    import notebook_kit as kit

    kit.altair_theme()
    return Path, alt, at, kit, mo, np, os, pd


@app.cell(hide_code=True)
def _(Path, at, mo, os):
    RUN = Path(mo.cli_args().get("run") or os.environ.get("RUN") or ".").expanduser()
    A = at.Analysis(RUN)
    mo.stop(not A.ready, mo.callout(mo.md(
        f"No prepared tables in `{RUN}/analysis/`. Build them first: "
        f"`python notebooks/build_tables.py --run {RUN}` (notebooks/README.md, step 1)."), kind="warn"))
    PERIODS = {f: A.periods(f) for f in ("quarterly", "monthly")}
    TILES = A.static("tiles")
    FOLLOWED = A.static("followed")
    return A, FOLLOWED, PERIODS, RUN, TILES


@app.cell(hide_code=True)
def _(A, PERIODS, RUN, mo):
    mo.md(f"""
    # 3. Distance from known mines, and how sure the model is

    Does the model behave differently far from where mining is known? This notebook measures
    how spread out the scores are (across chips, across the six models of the ensemble, and
    over time), whether chips the model calls "no mine" are near misses or clear zeros, and how
    both change with the distance to the nearest known mine.

    **Known mines:** {A.known_mines_text()}. Distance is measured from each chip's centre to the
    centre of the nearest known-mine chip, along the Earth's surface. Known mines come from
    years before the periods analysed here, so a detection does not make its own chip "known";
    when our own scoring replaces AMW's history (notebooks/analysis.yaml), this line says so.

    **Data.** Run `{RUN.name}`; quarters {len(PERIODS['quarterly'])}, months {len(PERIODS['monthly'])}.
    Histograms and rates cover every chip; per-chip measures use the sample (one chip in 1,000,
    the same chips in every period) and the followed chips.
    """)
    return


@app.cell(hide_code=True)
def _(at, mo):
    frequency = mo.ui.radio(options=["quarterly", "monthly"], value="quarterly", label="Frequency", inline=True)
    subregions = mo.ui.multiselect(options=[str(i) for i in range(1, 7)], value=[], label="Subregions (none = all)")
    rule = mo.ui.dropdown(options={v: k for k, v in at.RULES.items()}, value=at.RULES["det_raw"], label="Detection rule")
    mo.vstack([mo.hstack([frequency, subregions, rule], justify="start", gap=2),
               mo.md("<small>The rule sets the threshold that separates detected from not detected in section 2.</small>")])
    return frequency, rule, subregions


@app.cell(hide_code=True)
def _(kit):
    kit.glossary(["confidence", "vote_1 ... vote_6", "vote_sd", "dist_km", "known mine", "n_clear", "n_clear band",
                  "det_raw", "det_loose", "det_stringent", "det_andes", "subregion", "followed chip"])
    return


@app.cell(hide_code=True)
def _(A, FOLLOWED, PERIODS, at, frequency, subregions):
    FREQ = frequency.value
    TAGS = PERIODS[FREQ]
    SUBS = [int(s) for s in subregions.value] or list(range(1, 7))
    SUB_TEXT = "all subregions" if not subregions.value else "subregion" + ("s " if len(SUBS) > 1 else " ") + ", ".join(map(str, SUBS))
    bins = A.table(FREQ, "bins", TAGS)
    bins = bins[bins["subregion"].isin(SUBS) & (bins["score_bin"] >= 0)]
    bins["score_label"] = [at.score_label(i) for i in bins["score_bin"]]
    bins["dist_label"] = [at.DIST_LABELS[i] if i >= 0 else "no known mine" for i in bins["dist_band"]]
    sample = A.table(FREQ, "sample", TAGS)
    sample = sample[sample["subregion"].isin(SUBS) & sample["scored"]]
    hits = A.table(FREQ, "hits", TAGS)
    hits = hits[hits["chip_id"].isin(FOLLOWED.loc[FOLLOWED["subregion"].isin(SUBS), "chip_id"]) & hits["scored"]]
    SCORE_ORDER = [at.score_label(i) for i in range(len(at.SCORE_EDGES) - 1)]
    return FREQ, SCORE_ORDER, SUBS, SUB_TEXT, TAGS, bins, hits, sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How spread out are the model's scores?

    Three different spreads, each answering its own question: across chips (does the score
    distribution change between periods?), across the six votes (how unsure is the ensemble?),
    and over time (does a chip's score jump from period to period?). Almost every chip scores
    near 0, which dominates any plain standard deviation, so each figure also looks at the
    chips above 0.05.
    """)
    return


@app.cell(hide_code=True)
def _(SCORE_ORDER, SUB_TEXT, TAGS, alt, bins, kit):
    _g = bins.groupby(["period", "score_label"], observed=True)["chips"].sum().reset_index()
    _g["share"] = _g["chips"] / _g.groupby("period", observed=True)["chips"].transform("sum")
    _g["period"] = _g["period"].astype(str)
    _chart = alt.Chart(_g).mark_line(point=True).encode(
        x=alt.X("score_label:N", sort=SCORE_ORDER, title="confidence bin"),
        y=alt.Y("share:Q", scale=alt.Scale(type="log"), title="share of the period's scored chips (log scale)"),
        color=alt.Color("period:N", sort=TAGS, scale=alt.Scale(scheme="viridis"), title="period"),
        tooltip=["period", "score_label", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".2e")],
    ).properties(width="container", height=280)
    kit.figure(_chart, 1, "The score distribution of every period.",
               "The share of scored chips in each confidence bin, one line per period. The vertical axis is logarithmic: "
               "each step down is ten times fewer chips. Lines that lie on top of each other mean the periods have the "
               "same distribution.",
               mark="one bin in one period.", x="confidence bins; they are narrow near 0 and near the thresholds (0.4, 0.43, 0.55), "
                                               "so the bins are not of equal width.",
               y="share of the period's scored chips in the bin.", colour="period, from earliest (dark) to latest (light).",
               sample=f"every scored chip of {SUB_TEXT}.")
    return


@app.cell(hide_code=True)
def _(A, FREQ, SUBS, SUB_TEXT, TAGS, alt, at, bins, kit, np, pd):
    _rows = []
    _above = bins["score_bin"] >= at.SCORE_EDGES.index(0.05)
    for _tag, _g in bins.groupby("period", observed=True):
        for _label, _sel in (("all scored chips", _g), ("chips above 0.05", _g[_above.loc[_g.index]])):
            _n = _sel["chips"].sum()
            _mean = _sel["conf_sum"].sum() / _n if _n else np.nan
            _sd = np.sqrt(max(_sel["conf_sq"].sum() / _n - _mean ** 2, 0)) if _n else np.nan
            _rows += [{"period": str(_tag), "group": _label, "statistic": "mean", "value": _mean},
                      {"period": str(_tag), "group": _label, "statistic": "standard deviation", "value": _sd}]
        _h = A.table(FREQ, "hist", [str(_tag)])
        _h = _h[(_h["variable"] == "confidence") & (_h["subregion"].isin(SUBS) if len(SUBS) < 6 else _h["subregion"] == 0)]
        _h = _h.groupby(["left", "right"], as_index=False)["chips"].sum()
        _q = at.hist_quantiles(_h, (0.9, 0.99, 0.999))
        _rows += [{"period": str(_tag), "group": "all scored chips", "statistic": f"{int(k * 1000) / 10:g}th percentile", "value": v}
                  for k, v in _q.items()]
    _d = pd.DataFrame(_rows)
    _chart = alt.Chart(_d).mark_line(point=True).encode(
        x=alt.X("period:N", sort=TAGS, title=None), y=alt.Y("value:Q", title="confidence"),
        color=alt.Color("group:N", scale=alt.Scale(range=["#555555", kit.DETECTED]), title=None),
        tooltip=["period", "group", "statistic", alt.Tooltip("value:Q", format=".4f")],
    ).properties(width=250, height=170).facet(facet=alt.Facet("statistic:N", title=None, sort=[
        "mean", "standard deviation", "90th percentile", "99th percentile", "99.9th percentile"]), columns=3
    ).resolve_scale(y="independent")
    kit.figure(_chart, 2, "Spread across chips, period by period.",
               "Mean and standard deviation of confidence over all scored chips (grey) and over chips above 0.05 (blue), and "
               "three upper percentiles over all scored chips. The percentiles describe the tail where detections are.",
               mark="one period.", y="confidence (each panel has its own scale).",
               colour="grey: all scored chips; blue: chips with confidence above 0.05.",
               sample=f"every scored chip of {SUB_TEXT}. Percentiles come from histograms with bins 0.01 wide.")
    return


@app.cell(hide_code=True)
def _(SCORE_ORDER, SUB_TEXT, alt, at, bins, kit):
    _g = bins.groupby(["score_label", "dist_label"], observed=True)[["chips", "vote_sd_sum"]].sum().reset_index()
    _g["mean vote_sd"] = _g["vote_sd_sum"] / _g["chips"]
    _g = _g[_g["chips"] >= 30]
    _chart = alt.Chart(_g).mark_line(point=True).encode(
        x=alt.X("score_label:N", sort=SCORE_ORDER, title="confidence bin"),
        y=alt.Y("mean vote_sd:Q", title="mean vote_sd (disagreement of the six models)"),
        color=alt.Color("dist_label:N", sort=at.DIST_LABELS, scale=alt.Scale(scheme="plasma"), title="distance to the nearest known mine"),
        tooltip=["score_label", "dist_label", alt.Tooltip("chips:Q", format=","), alt.Tooltip("mean vote_sd:Q", format=".4f")],
    ).properties(width="container", height=280)
    kit.figure(_chart, 3, "Disagreement among the six models, by score and distance.",
               "The mean standard deviation of the six votes for chips in each confidence bin, one line per distance band. "
               "Disagreement is low where all six say clearly no or clearly yes and highest in between. If lines for "
               "far-away chips lie above the others at the same score, the ensemble is less sure far from known mines.",
               mark="one confidence bin and distance band, if it holds at least 30 chip-periods.",
               x="confidence bins (not of equal width).", y="mean of vote_sd over the chip-periods in the bin.",
               colour="distance from the chip to the nearest known mine.",
               sample=f"every scored chip-period of {SUB_TEXT}, all periods of the chosen frequency pooled.")
    return


@app.cell(hide_code=True)
def _(FOLLOWED, alt, hits, kit, np, pd, rule, sample):
    def _per_chip(d):
        g = d.groupby("chip_id").agg(periods=("confidence", "size"), mean_conf=("confidence", "mean"),
                                     sd_time=("confidence", "std"), mean_vote_sd=("vote_sd", "mean"))
        return g[g["periods"] >= 3]

    _known = set(FOLLOWED.loc[FOLLOWED["known"], "chip_id"])
    _ever = set(hits.loc[hits[rule.value], "chip_id"])
    per_chip = pd.concat([
        _per_chip(sample).assign(group="typical chip (sample)"),
        _per_chip(hits[hits["chip_id"].isin(_ever)]).assign(group="detected at least once"),
        _per_chip(hits[hits["chip_id"].isin(_known)]).assign(group="known mine"),
    ]).reset_index()
    _d = per_chip.assign(log_sd=np.log10(per_chip["sd_time"].clip(lower=1e-4)))
    _chart = alt.Chart(_d).transform_density("log_sd", groupby=["group"], as_=["log_sd", "density"], extent=[-4, 0]).mark_line().encode(
        x=alt.X("log_sd:Q", title="standard deviation of the chip's confidence over time (log10; -2 = 0.01, -1 = 0.1)"),
        y=alt.Y("density:Q", title="density"),
        color=alt.Color("group:N", scale=alt.Scale(domain=["typical chip (sample)", "detected at least once", "known mine"],
                                                   range=["#999999", kit.DETECTED, kit.AMW]), title=None),
    ).properties(width="container", height=240)
    kit.figure(_chart, 4, "How much a chip's score moves over time.",
               "For each chip, the standard deviation of its confidence across the periods, as a smoothed histogram per group. "
               "Typical chips stay near 0 and barely move; mines move more, because their score is high when they are seen and "
               "low when they are not.",
               mark="one line per group: a kernel density of the chips' values (area 1 per group).",
               x="log10 of the over-time standard deviation; values under 0.0001 are set to 0.0001.",
               y="density.", colour="group of chips.",
               sample="chips scored in at least 3 periods of the chosen frequency; the typical chips are the sample of one in 1,000.")
    return (per_chip,)


@app.cell(hide_code=True)
def _(alt, kit, np, pd, per_chip):
    _d = per_chip[per_chip["group"] != "typical chip (sample)"].drop_duplicates("chip_id").dropna()
    _rows = []
    _names = {"mean_conf": "mean confidence", "sd_time": "sd over time", "mean_vote_sd": "disagreement (vote_sd)"}
    for _a, _b in (("mean_conf", "sd_time"), ("mean_conf", "mean_vote_sd"), ("sd_time", "mean_vote_sd")):
        _rows.append({"pair": f"{_names[_a]} vs {_names[_b]}", "Spearman rank correlation": _d[_a].rank().corr(_d[_b].rank())})
    _corr = pd.DataFrame(_rows)
    _h = _d.assign(x=(np.floor(_d["mean_vote_sd"] / 0.01) * 0.01).round(2), y=(np.floor(_d["sd_time"] / 0.02) * 0.02).round(2))
    _h = _h.groupby(["x", "y"]).agg(chips=("chip_id", "size"), mean_conf=("mean_conf", "mean")).reset_index()
    _heat = alt.Chart(_h).mark_rect().encode(
        x=alt.X("x:Q", title="mean vote_sd over the chip's periods (disagreement)"),
        x2=alt.X2("x2:Q"), y=alt.Y("y:Q", title="standard deviation over time"),
        y2="y2:Q",
        color=alt.Color("mean_conf:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="mean confidence of the chips"),
        opacity=alt.Opacity("chips:Q", scale=alt.Scale(type="log", range=[0.25, 1]), title="chips"),
        tooltip=[alt.Tooltip("chips:Q", format=","), alt.Tooltip("mean_conf:Q", format=".2f")],
    ).transform_calculate(x2="datum.x + 0.01", y2="datum.y + 0.02").properties(width="container", height=300)
    _bars = alt.Chart(_corr).mark_bar(color=kit.DETECTED).encode(
        y=alt.Y("pair:N", title=None), x=alt.X("Spearman rank correlation:Q", scale=alt.Scale(domain=[-1, 1])),
        tooltip=[alt.Tooltip("Spearman rank correlation:Q", format=".3f")]).properties(width="container", height=90)
    import marimo as _mo
    _mo.vstack([
        kit.figure(_heat, 5, "The spreads in combination.",
                   "Each cell groups chips by their disagreement among the six models (horizontal) and by how much their score "
                   "moves over time (vertical). Colour shows the chips' mean confidence; faint cells hold few chips.",
                   mark="one cell of 0.01 (horizontal) by 0.02 (vertical).",
                   x="the chip's mean vote_sd over its scored periods.", y="standard deviation of the chip's confidence over time.",
                   colour="mean over the cell's chips of their mean confidence; opacity: number of chips (log scale).",
                   sample="followed chips detected at least once or known mines, scored in at least 3 periods."),
        kit.figure(_bars, 6, "Rank correlations between the three chip-level measures.",
                   "Spearman's rank correlation for each pair of mean confidence, over-time standard deviation and mean "
                   "disagreement: +1 = same order, 0 = unrelated, -1 = opposite order.",
                   mark="one pair of measures.", sample="the chips of Figure 5."),
    ])
    return


@app.cell(hide_code=True)
def _(at, mo, rule):
    _t = at.THRESHOLD[rule.value]
    _below = [e for e in at.SCORE_EDGES if e < _t]
    NEAR_FROM = max(e for e in _below if e <= _t - 0.1 + 1e-9) if any(e <= _t - 0.1 + 1e-9 for e in _below) else 0
    mo.md(f"""
    ## 2. Chips called "no mine": near misses or clear zeros?

    With the {at.RULES[rule.value]} rule, a chip is not detected below {_t:g}. A *near miss* here is a
    non-detected chip scoring from {NEAR_FROM:g} up to the threshold (the bin edges closest to 0.1 below it).
    """)
    return (NEAR_FROM,)


@app.cell(hide_code=True)
def _(SCORE_ORDER, SUB_TEXT, alt, at, bins, kit, rule):
    _t = at.THRESHOLD[rule.value]
    _nd = bins[[at.SCORE_EDGES[i + 1] <= _t + 1e-9 for i in bins["score_bin"]]]
    _g = _nd.groupby(["dist_label", "score_label"], observed=True)["chips"].sum().reset_index()
    _g["share"] = _g["chips"] / _g.groupby("dist_label")["chips"].transform("sum")
    _order = [s for s in SCORE_ORDER if s in set(_g["score_label"])]
    _chart = alt.Chart(_g).mark_line(point=True).encode(
        x=alt.X("score_label:N", sort=_order, title="confidence bin (below the threshold)"),
        y=alt.Y("share:Q", scale=alt.Scale(type="log"), title="share of the band's non-detected chips (log scale)"),
        color=alt.Color("dist_label:N", sort=at.DIST_LABELS, scale=alt.Scale(scheme="plasma"), title="distance to the nearest known mine"),
        tooltip=["dist_label", "score_label", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".2e")],
    ).properties(width="container", height=280)
    kit.figure(_chart, 7, "Where non-detected chips score, by distance to known mines.",
               "Of the chips not detected in a distance band, the share in each confidence bin below the threshold. A line that "
               "stays high toward the right (near the threshold) means many near misses in that band; a line that drops "
               "steeply means the band's non-detections are clear zeros.",
               mark="one bin and distance band.", x="confidence bins below the rule's threshold (not of equal width).",
               y="share of the band's non-detected chip-periods in the bin.", colour="distance band.",
               sample=f"every scored, non-detected chip-period of {SUB_TEXT}, all periods of the chosen frequency pooled. "
                      f"'Non-detected' here means below {_t:g}; for the loose and stringent rules a few chips above the "
                      f"threshold are also not detected because they are isolated, and are left out.")
    return


@app.cell(hide_code=True)
def _(NEAR_FROM, SUB_TEXT, TAGS, alt, at, bins, kit, np, rule):
    _t = at.THRESHOLD[rule.value]
    _b = bins[bins["n_clear_band"] >= 0].copy()
    _b["near"] = [(at.SCORE_EDGES[i] >= NEAR_FROM - 1e-9) and (at.SCORE_EDGES[i + 1] <= _t + 1e-9) for i in _b["score_bin"]]
    _b["nd"] = [at.SCORE_EDGES[i + 1] <= _t + 1e-9 for i in _b["score_bin"]]
    _b = _b[_b["nd"]]
    _b["near_chips"] = _b["chips"].where(_b["near"], 0)
    _g = _b.groupby(["dist_label", "n_clear_band"], observed=True).agg(
        near=("near_chips", "sum"), chips=("chips", "sum")).reset_index()
    _g = _g[_g["chips"] > 0]
    _g["per_1000"] = 1000 * _g["near"] / _g["chips"]
    _g["cloud"] = [at.NCLEAR_LABELS[i] for i in _g["n_clear_band"]]
    _base = alt.Chart(_g).encode(
        x=alt.X("cloud:N", sort=at.NCLEAR_LABELS, title="n_clear band", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("dist_label:N", sort=at.DIST_LABELS, title="distance to the nearest known mine"))
    _chart = alt.layer(
        _base.mark_rect().encode(
            color=alt.Color("per_1000:Q", scale=alt.Scale(scheme="viridis", type="symlog"),
                            title="near misses per 1,000 non-detected chips"),
            tooltip=["dist_label", "cloud", alt.Tooltip("near:Q", format=","), alt.Tooltip("chips:Q", format=","),
                     alt.Tooltip("per_1000:Q", format=".2f")]),
        _base.mark_text(fontSize=12).encode(
            text=alt.Text("per_1000:Q", format=".1f"),
            color=alt.condition("datum.per_1000 > 20", alt.value("black"), alt.value("white"))),
    ).properties(width="container", height=230)
    kit.figure(_chart, 8, "Near misses by distance and cloud.",
               f"Of the non-detected chip-periods in each distance band and n_clear band, how many per 1,000 were near misses "
               f"(confidence from {NEAR_FROM:g} to {_t:g}). Near known mines and under good views, near misses are expected to be "
               f"most common; near misses far away or under clouds are the cases to look at.",
               mark="one distance band and n_clear band; the number in the cell is its value.",
               colour="near misses per 1,000 non-detected chip-periods (compressed scale, so small values stay visible).",
               sample=f"every scored, non-detected chip-period of {SUB_TEXT}, all periods of the chosen frequency pooled.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Distance and confidence
    """)
    return


@app.cell(hide_code=True)
def _(SUB_TEXT, alt, at, bins, kit, np, rule):
    _r = {"det_raw": "raw", "det_loose": "loose", "det_stringent": "stringent", "det_andes": "andes"}[rule.value]
    _all = bins.assign(sub="all")
    _g = _all.groupby(["dist_label"], observed=True)[["chips", "conf_sum", "conf_sq", _r]].sum().reset_index()
    _g["mean"] = _g["conf_sum"] / _g["chips"]
    _g["se"] = np.sqrt((_g["conf_sq"] / _g["chips"] - _g["mean"] ** 2).clip(lower=0) / _g["chips"])
    _g["low"], _g["high"] = _g["mean"] - 1.96 * _g["se"], _g["mean"] + 1.96 * _g["se"]
    _g["det_per_1000"] = 1000 * _g[_r] / _g["chips"]
    _lo, _hi = at.wilson(_g[_r], _g["chips"])
    _g["det_low"], _g["det_high"] = 1000 * _lo, 1000 * _hi
    _x = alt.X("dist_label:N", sort=at.DIST_LABELS, title="distance to the nearest known mine", axis=alt.Axis(labelAngle=0))
    _c1 = alt.layer(alt.Chart(_g).mark_rule(color="black").encode(x=_x, y=alt.Y("low:Q", title="mean confidence"), y2="high:Q"),
                    alt.Chart(_g).mark_point(filled=True, size=80, color=kit.DETECTED).encode(
                        x=_x, y="mean:Q", tooltip=["dist_label", alt.Tooltip("chips:Q", format=","), alt.Tooltip("mean:Q", format=".4f")])
                    ).properties(width=380, height=230)
    _c2 = alt.layer(alt.Chart(_g).mark_rule(color="black").encode(x=_x, y=alt.Y("det_low:Q", title=f"{_r} detections per 1,000 chips",
                                                                               scale=alt.Scale(type="symlog")), y2="det_high:Q"),
                    alt.Chart(_g).mark_point(filled=True, size=80, color=kit.AMW).encode(
                        x=_x, y="det_per_1000:Q", tooltip=["dist_label", alt.Tooltip("det_per_1000:Q", format=".2f")])
                    ).properties(width=380, height=230)
    kit.figure(alt.hconcat(_c1, _c2), 9, "Confidence and detections fall with distance from known mines.",
               "Left: mean confidence of the chips in each distance band. Right: the rule's detections per 1,000 chips in the band.",
               mark="one distance band; the black line is the 95% interval (left: mean ± 1.96 standard errors; right: Wilson interval).",
               y="left: mean confidence; right: detections per 1,000 scored chip-periods (scale linear near 0, logarithmic above 1).",
               sample=f"every scored chip-period of {SUB_TEXT}, all periods of the chosen frequency pooled.")
    return


@app.cell(hide_code=True)
def _(alt, at, kit, np, pd, sample):
    _s = sample.dropna(subset=["dist_km", "confidence"])
    _rows = [{"group": "all chips", "rho": _s["dist_km"].rank().corr(_s["confidence"].rank()), "n": len(_s)}]
    for _i, _lab in enumerate(at.NCLEAR_LABELS):
        _g = _s[[at.NCLEAR_EDGES[_i] <= v < at.NCLEAR_EDGES[_i + 1] for v in _s["n_clear"]]]
        if len(_g) >= 50:
            _rows.append({"group": f"n_clear {_lab}", "rho": _g["dist_km"].rank().corr(_g["confidence"].rank()), "n": len(_g)})
    _d = pd.DataFrame(_rows)
    _s = _s.assign(logd=np.log10(_s["dist_km"].clip(lower=0.1)))
    _bins = at.rate_by_bins(_s.assign(_hi=_s["confidence"] > 0.05), "logd", "_hi", 20)
    _mean = _s.groupby(pd.qcut(_s["logd"], 20, duplicates="drop"), observed=True).agg(x=("logd", "median"), mean_conf=("confidence", "mean")).reset_index(drop=True)
    _c1 = alt.Chart(_mean).mark_line(point=True, color=kit.DETECTED).encode(
        x=alt.X("x:Q", title="distance to the nearest known mine (log10 km; 0 = 1 km, 1 = 10 km, 2 = 100 km)"),
        y=alt.Y("mean_conf:Q", title="mean confidence"), tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("mean_conf:Q", format=".4f")]
    ).properties(width=420, height=230)
    _c2 = alt.Chart(_d).mark_bar(color=kit.DETECTED).encode(
        y=alt.Y("group:N", sort=list(_d["group"]), title=None), x=alt.X("rho:Q", title="Spearman rank correlation", scale=alt.Scale(domain=[-1, 0.2])),
        tooltip=["group", alt.Tooltip("rho:Q", format=".3f"), alt.Tooltip("n:Q", format=",")]).properties(width=300, height=230)
    kit.figure(alt.hconcat(_c1, _c2), 10, "Distance against confidence, chip by chip.",
               "Left: the sampled chip-periods split into 20 groups of equal size by distance, with the mean confidence of each "
               "group. Right: the rank correlation between distance and confidence, over all sampled chip-periods and within each "
               "n_clear band (so that cloud cover, which differs between near and far areas, is held fixed).",
               mark="left: one twentieth of the sampled chip-periods; right: one group.",
               x="left: log10 of the distance in km (distances under 0.1 km are set to 0.1).",
               sample="the sample (one chip in 1,000, the same chips in every period), every period of the chosen frequency.")
    return


@app.cell(hide_code=True)
def _(A, FREQ, NEAR_FROM, SUBS, TAGS, TILES, at, kit, mo, rule):
    _pt = A.table(FREQ, "tiles", TAGS).merge(TILES[["tile", "subregion"]], on="tile")
    _pt = _pt[_pt["subregion"].isin(SUBS)].drop(columns="subregion")
    _cells = at.to_cells(_pt, TILES, by=["period"])
    layer = mo.ui.dropdown(options={"near misses (0.3 to 0.4) per 1,000 chips": "near_raw_per_1000",
                                    "mean vote_sd (disagreement)": "vote_sd_mean",
                                    "mean confidence": "confidence_mean"},
                           value="near misses (0.3 to 0.4) per 1,000 chips", label="Left map")
    CELLS3 = _cells
    layer
    return CELLS3, layer


@app.cell(hide_code=True)
def _(CELLS3, TAGS, kit, layer, rule):
    _r = {"det_raw": "raw", "det_loose": "loose", "det_stringent": "stringent", "det_andes": "andes"}[rule.value]
    _fig = kit.period_map(CELLS3, TAGS, [
        {"column": layer.value, "title": layer.selected_key, "colorscale": "Magma", "reverse": True},
        {"column": f"{_r}_per_1000", "title": f"{_r} detections per 1,000 chips", "colorscale": "Viridis"}])
    kit.figure(_fig, 11, "Near misses, disagreement and detections on the map.",
               "Left: the layer chosen above; right: the rule's detections per 1,000 chips. Cells with near misses but no "
               "detections mark where the model almost finds mining.",
               mark="one cell of 0.25 degrees, as in notebook 1.",
               colour="left: the chosen layer (near misses always use the raw threshold: 0.3 to 0.4); right: detections per 1,000 "
                      "scored chips. Both scales stop at the 98th percentile.",
               sample="every scored chip of the selected subregions, at the chosen frequency; slider and play button step through periods.")
    return


@app.cell(hide_code=True)
def _(FOLLOWED, hits, kit, mo, np, pd, rule):
    _far = hits[hits[rule.value]].merge(FOLLOWED[["chip_id", "lon", "lat", "subregion", "country", "dist_km"]], on="chip_id")
    _far = _far[_far["dist_km"] > 20]
    _g = _far.groupby("chip_id").agg(periods_detected=("period", "size"), first=("period", "min"), last=("period", "max"),
                                     max_confidence=("confidence", "max"), mean_vote_sd=("vote_sd", "mean"),
                                     mean_n_clear=("n_clear", "mean"), dist_km=("dist_km", "first"), lon=("lon", "first"),
                                     lat=("lat", "first"), subregion=("subregion", "first"), country=("country", "first")).reset_index()
    _g["satellite view"] = [f"https://www.google.com/maps/@{la:.5f},{lo:.5f},16z/data=!3m1!1e3" for la, lo in zip(_g["lat"], _g["lon"])]
    _g = _g.sort_values(["periods_detected", "max_confidence"], ascending=False).round(3)
    _g["first"], _g["last"] = _g["first"].astype(str), _g["last"].astype(str)
    mo.vstack([
        mo.md(f"### Detections far from any known mine\n\n{len(_g):,} chips were detected at least once more than 20 km from "
              f"the nearest known mine. They are either new mining fronts or false alarms; sorted by how often they were "
              f"detected. Download the table with the button under it."),
        mo.ui.table(_g, selection=None, page_size=15, show_column_summaries=False),
        kit.caption(12, "Far-away detections.", "One row per chip detected more than 20 km from the nearest known mine.",
                    sample="followed chips, chosen frequency and rule; columns as in the variable list, plus the number of "
                           "periods detected, the first and last of them, and a link to the satellite view."),
    ])
    return


@app.cell(hide_code=True)
def _(A, mo):
    mo.md(f"""
    ---
    <small>Built from `{A.dir}` ({A.log.get('built_utc', '?')}).</small>
    """)
    return


if __name__ == "__main__":
    app.run()
