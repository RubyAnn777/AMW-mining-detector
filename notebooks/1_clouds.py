import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="1 Clouds")


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
    return Path, REPO, alt, at, kit, mo, np, os, pd


@app.cell(hide_code=True)
def _(Path, at, mo, os):
    RUN = Path(mo.cli_args().get("run") or os.environ.get("RUN") or ".").expanduser()
    A = at.Analysis(RUN)
    mo.stop(not A.ready, mo.callout(mo.md(
        f"No prepared tables in `{RUN}/analysis/`. Build them first: "
        f"`python notebooks/build_tables.py --run {RUN}` (notebooks/README.md, step 1)."), kind="warn"))
    PERIODS = {f: A.periods(f) for f in ("quarterly", "monthly")}
    TILES = A.static("tiles")
    return A, PERIODS, RUN, TILES


@app.cell(hide_code=True)
def _(A, PERIODS, RUN, mo):
    def _span(tags):
        return f"{tags[0]} to {tags[-1]} ({len(tags)})" if tags else "none"

    mo.md(f"""
    # 1. Clouds: how well each period was seen, and what it does to detections

    The model scores one picture per chip and period: the median of the period's clear
    Sentinel-2 images (an image is clear where Cloud Score+ is at least 0.6). Where a pixel has
    no clear image, the picture is black there. This notebook measures how much of the basin was
    seen in each period and whether the periods in which a mine was missed were the cloudy ones.

    **Data.** Run `{RUN.name}`; quarters {_span(PERIODS['quarterly'])}; months {_span(PERIODS['monthly'])}.
    Tables built {A.log.get('built_utc', '?')[:16].replace('T', ' ')} UTC by `notebooks/build_tables.py`.
    Counts, shares and maps cover every chip; sections 3 and 4 follow chips that are mines.
    Known mines: {A.known_mines_text()}.
    """)
    return


@app.cell(hide_code=True)
def _(at, mo):
    frequency = mo.ui.radio(options=["quarterly", "monthly"], value="quarterly", label="Frequency", inline=True)
    subregions = mo.ui.multiselect(options=[str(i) for i in range(1, 7)], value=[], label="Subregions (none = all)")
    rule = mo.ui.dropdown(options={v: k for k, v in at.RULES.items()}, value=at.RULES["det_loose"], label="Detection rule")
    mo.vstack([mo.hstack([frequency, subregions, rule], justify="start", gap=2),
               mo.md("<small>These three settings apply to every figure below unless its caption says otherwise.</small>")])
    return frequency, rule, subregions


@app.cell(hide_code=True)
def _(kit):
    kit.glossary(["n_obs", "n_clear", "n_clear_median", "empty_share", "thin_share", "cs_mean", "cirrus_mean",
                  "confidence", "det_raw", "det_loose", "det_stringent", "det_andes", "subregion",
                  "n_clear band", "share_n_clear_0", "share_n_clear_under_1", "detections per 1,000 chips",
                  "known mine", "followed chip", "detected / missed", "AUC"])
    return


@app.cell(hide_code=True)
def _(PERIODS, frequency, subregions):
    FREQ = frequency.value
    TAGS = PERIODS[FREQ]
    SUBS = [int(s) for s in subregions.value] or list(range(1, 7))
    SUB_TEXT = "all subregions" if len(SUBS) == 6 or not subregions.value else \
        "subregion" + ("s " if len(SUBS) > 1 else " ") + ", ".join(map(str, SUBS))
    return FREQ, SUBS, SUB_TEXT, TAGS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How cloudy was each period?
    """)
    return


@app.cell(hide_code=True)
def _(A, PERIODS, SUBS, SUB_TEXT, alt, at, kit, pd):
    def _over_time():
        rows = []
        for f, tags in PERIODS.items():
            h = A.table(f, "hist", tags)
            if h.empty:
                continue
            h = h[h["subregion"].isin(SUBS)] if len(SUBS) < 6 else h[h["subregion"] == 0]
            for tag, g in h.groupby("period", observed=True):
                nc = g[g["variable"] == "n_clear"].groupby(["left", "right"], as_index=False)["chips"].sum()
                em = g[g["variable"] == "empty_share"].groupby(["left", "right"], as_index=False)["chips"].sum()
                start = at.period_middle(str(tag))
                rows += [{"frequency": f, "period": str(tag), "date": start, "measure": "median n_clear",
                          "value": at.hist_quantiles(nc, (0.5,))[0.5]},
                         {"frequency": f, "period": str(tag), "date": start, "measure": "share of chips with n_clear under 1",
                          "value": at.hist_share(nc, 1.0)},
                         {"frequency": f, "period": str(tag), "date": start, "measure": "share of chips more than half black",
                          "value": 1 - at.hist_share(em, 0.5 + 1e-9) if em["chips"].sum() else None}]
        return pd.DataFrame(rows)

    over_time = _over_time()
    _chart = alt.Chart(over_time).mark_line(point=True).encode(
        x=alt.X("date:T", title="middle of the period", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("frequency:N", scale=alt.Scale(domain=["monthly", "quarterly"], range=[kit.MONTHLY, kit.QUARTERLY]),
                        title=None),
        tooltip=["period", "measure", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=230, height=200).facet(
        column=alt.Column("measure:N", title=None, sort=["median n_clear", "share of chips with n_clear under 1",
                                                         "share of chips more than half black"])
    ).resolve_scale(y="independent")
    kit.figure(_chart, 1, "Cloud cover over time.",
               "Three measures of how well each period was seen, for months and quarters on one time axis. "
               "A month has about a third of a quarter's images, so its n_clear is lower by construction; "
               "compare months with months and quarters with quarters.",
               mark="one period, placed at the middle of the period.",
               y="left: median over chips of n_clear (clear images per pixel); middle: share of scored chips with "
                 "n_clear under 1; right: share of scored chips with empty_share above 0.5.",
               colour="frequency.",
               sample=f"every scored chip of {SUB_TEXT}, every period built.")
    return


@app.cell(hide_code=True)
def _(A, FREQ, SUBS, SUB_TEXT, TAGS, alt, at, kit, pd):
    def _boxes():
        h = A.table(FREQ, "hist", TAGS)
        h = h[h["variable"] == "n_clear"]
        h = h[h["subregion"].isin(SUBS)] if len(SUBS) < 6 else h[h["subregion"] == 0]
        rows = []
        for tag, g in h.groupby("period", observed=True):
            g = g.groupby(["left", "right"], as_index=False)["chips"].sum()
            q = at.hist_quantiles(g)
            rows.append({"period": str(tag), "p5": q[0.05], "p25": q[0.25], "p50": q[0.5], "p75": q[0.75], "p95": q[0.95],
                         "chips": int(g["chips"].sum())})
        return pd.DataFrame(rows)

    _b = _boxes()
    _base = alt.Chart(_b).encode(x=alt.X("period:N", sort=TAGS, title=None))
    _y = alt.Scale(type="symlog", constant=1)
    _chart = alt.layer(
        _base.mark_rule(color="#888").encode(y=alt.Y("p5:Q", scale=_y, title="n_clear (clear images per pixel)"), y2="p95:Q"),
        _base.mark_bar(size=18, color=kit.QUARTERLY if FREQ == "quarterly" else kit.MONTHLY, opacity=0.8).encode(y="p25:Q", y2="p75:Q"),
        _base.mark_tick(color="black", size=18, thickness=2).encode(y="p50:Q"),
    ).encode(tooltip=["period", alt.Tooltip("p5:Q", format=".2f"), alt.Tooltip("p25:Q", format=".2f"),
                      alt.Tooltip("p50:Q", format=".2f"), alt.Tooltip("p75:Q", format=".2f"),
                      alt.Tooltip("p95:Q", format=".2f"), alt.Tooltip("chips:Q", format=",")]
    ).properties(width="container", height=260)
    kit.figure(_chart, 2, f"Distribution of n_clear across chips, per {FREQ[:-2]}.",
               "How the number of clear images per pixel is spread over the basin's chips in each period. "
               "A low box means most of the basin was poorly seen; a tall box means some areas were seen well and others not.",
               mark="one period. Bar: middle half of chips (25th to 75th percentile); black tick: median; "
                    "thin line: 5th to 95th percentile.",
               y="n_clear on a scale that is linear near 0 and logarithmic above 1, so both cloudy and clear periods are readable.",
               sample=f"every scored chip of {SUB_TEXT}. Percentiles are read from histograms with bins 0.5 wide.")
    return


@app.cell(hide_code=True)
def _(mo):
    split = mo.ui.radio(options=["subregion", "country"], value="subregion", label="Split by", inline=True)
    cloud_measure = mo.ui.dropdown(
        options={"share of chips with n_clear under 1": "share_n_clear_under_1",
                 "share of chips with n_clear = 0": "share_n_clear_0",
                 "mean n_clear": "n_clear_mean",
                 "share of chips more than half black": "share_empty_over_half",
                 "mean empty_share": "empty_share_mean",
                 "mean cs_mean": "cs_mean_mean"},
        value="share of chips with n_clear under 1", label="Cloud measure (Figures 3 and 4)")
    mo.hstack([split, cloud_measure], justify="start", gap=2)
    return cloud_measure, split


@app.cell(hide_code=True)
def _(A, FREQ, SUBS, TAGS, TILES):
    period_tiles = A.table(FREQ, "tiles", TAGS)
    period_tiles = period_tiles.merge(TILES[["tile", "subregion", "country"]], on="tile")
    period_tiles = period_tiles[period_tiles["subregion"].isin(SUBS)]
    return (period_tiles,)


@app.cell(hide_code=True)
def _(TAGS, alt, cloud_measure, kit, np, pd, period_tiles, split):
    _by = split.value
    _m = cloud_measure.value
    _t = period_tiles.assign(_w=period_tiles[_m].astype(float) * period_tiles["scored"])
    _g = _t.groupby([_by, "period"], observed=True).agg(w=("_w", "sum"), scored=("scored", "sum")).reset_index()
    _g["value"] = _g["w"] / _g["scored"].replace(0, np.nan)
    _g[_by] = _g[_by].astype(str)
    _g["period"] = _g["period"].astype(str)
    _chart = alt.Chart(_g).mark_rect().encode(
        x=alt.X("period:N", sort=TAGS, title=None),
        y=alt.Y(f"{_by}:N", title=_by),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blues", reverse=_m in ("n_clear_mean", "cs_mean_mean")),
                        title=cloud_measure.selected_key),
        tooltip=[_by, "period", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("scored:Q", format=",", title="chips scored")],
    ).properties(width="container", height=max(120, 26 * _g[_by].nunique()))
    kit.figure(_chart, 3, f"Cloud cover by {_by} and period.",
               f"The chosen cloud measure for each {_by} in each period. Darker = cloudier "
               f"(for mean n_clear and mean cs_mean the scale is reversed so that darker still means cloudier).",
               mark=f"one {_by} in one period.",
               colour=f"{cloud_measure.selected_key}, over the scored chips of the {_by}.",
               sample="every scored chip of the selected subregions; countries are taken from each tile's centre "
                      "(Natural Earth 1:110m outlines).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Where it was cloudy, period by period

    The map below has its own time slider and play button, which keep working in the static
    HTML version. The left map shows the chosen cloud measure, the right map the detections, in
    cells of 0.25 degrees (about 28 km). Switch the frequency at the top to see the months.
    """)
    return


@app.cell(hide_code=True)
def _(TAGS, TILES, at, cloud_measure, kit, period_tiles, rule):
    _rule = {"det_raw": "raw", "det_loose": "loose", "det_stringent": "stringent", "det_andes": "andes"}[rule.value]
    cells = at.to_cells(period_tiles.drop(columns=["subregion", "country"]), TILES, by=["period"])
    map_fig = kit.period_map(cells, TAGS, [
        {"column": cloud_measure.value, "title": cloud_measure.selected_key, "colorscale": "Blues",
         "reverse": cloud_measure.value in ("n_clear_mean", "cs_mean_mean")},
        {"column": f"{_rule}_per_1000", "title": f"{_rule} detections per 1,000 chips", "colorscale": "Viridis"}])
    kit.figure(map_fig, 4, "Cloud cover and detections on the map, by period.",
               "Left: the chosen cloud measure per cell. Right: the chosen rule's detections per 1,000 scored chips "
               "per cell. Move the slider (or press play) to step through the periods.",
               mark="one cell of 0.25 × 0.25 degrees (about 28 km a side), drawn if it holds at least one tile "
                    "(a tile's centre decides its cell).",
               colour="left: the cloud measure chosen above Figure 3 (darker = cloudier); right: detections per 1,000 "
                      "scored chips. Both scales stop at the 98th percentile over all periods and cells so that a few "
                      "extreme cells do not wash out the rest; values above it get the darkest colour.",
               sample="every scored chip of the selected subregions, at the chosen frequency.",
               note="the background map needs an internet connection; the cells draw without it.")
    return (cells,)


@app.cell(hide_code=True)
def _(PERIODS, TAGS, mo):
    _q = [q for q in PERIODS["quarterly"] if all(m in PERIODS["monthly"] for m in
                                               [f"{q[:4]}-{(int(q[-1]) - 1) * 3 + i:02d}" for i in (1, 2, 3)])]
    quarter_pick = mo.ui.dropdown(options=_q, value=_q[-1] if _q else None, label="Quarter (Figure 5)")
    quarter_pick if _q else mo.md("*Figure 5 needs a quarter whose three months are built.*")
    return (quarter_pick,)


@app.cell(hide_code=True)
def _(A, SUBS, TILES, alt, at, cloud_measure, kit, mo, quarter_pick):
    mo.stop(not quarter_pick.value)
    _q = quarter_pick.value
    _tags = [("quarterly", _q)] + [("monthly", m) for m in at.months_of(_q)]
    _parts = []
    for _f, _t in _tags:
        _pt = A.table(_f, "tiles", [_t]).merge(TILES[["tile", "subregion"]], on="tile")
        _pt = _pt[_pt["subregion"].isin(SUBS)].drop(columns="subregion")
        _c = at.to_cells(_pt, TILES)
        _c["panel"] = f"{_t} ({'quarter' if _f == 'quarterly' else 'month'})"
        _parts.append(_c)
    import pandas as _pd
    _m = cloud_measure.value
    _cells = _pd.concat(_parts, ignore_index=True)[["cell_lon", "cell_lat", "panel", _m]].round(4)
    _chart = alt.Chart(_cells).mark_square(size=40).encode(
        x=alt.X("cell_lon:Q", title="longitude", scale=alt.Scale(zero=False)),
        y=alt.Y("cell_lat:Q", title="latitude", scale=alt.Scale(zero=False)),
        color=alt.Color(f"{_m}:Q", scale=alt.Scale(scheme="blues", reverse=_m in ("n_clear_mean", "cs_mean_mean")),
                        title=cloud_measure.selected_key),
        tooltip=[alt.Tooltip(f"{_m}:Q", format=".3f"), "panel"],
    ).properties(width=200, height=200).facet(facet=alt.Facet("panel:N", title=None,
                                                              sort=[p for p in _cells["panel"].unique()]), columns=4)
    kit.figure(_chart, 5, f"A quarter next to its three months: {_q}.",
               "The same cloud measure for the quarter and for each of its months, side by side. A month only has "
               "its own images, so it is cloudier than its quarter; the quarter's picture is the median of all three months' clear images.",
               mark="one 0.25 degree cell.", x="longitude of the cell centre.", y="latitude of the cell centre.",
               colour=f"{cloud_measure.selected_key} (darker = cloudier). The four panels share one scale.",
               sample="every scored chip of the selected subregions.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Are fewer mines detected where it is cloudy?

    Detection rates by how well a chip was seen, across all chips. A warning before reading
    Figures 6 and 7: clouds and mining are not spread the same way over the basin, so these
    rates mix "cloudy" with "where the mines are". Section 4 removes that by comparing each mine
    only with itself.
    """)
    return


@app.cell(hide_code=True)
def _(A, FREQ, SUBS, TAGS):
    bins = A.table(FREQ, "bins", TAGS)
    bins = bins[bins["subregion"].isin(SUBS)]
    return (bins,)


@app.cell(hide_code=True)
def _(TAGS, alt, at, bins, kit, np, rule):
    _r = {"det_raw": "raw", "det_loose": "loose", "det_stringent": "stringent", "det_andes": "andes"}[rule.value]

    def _rates(band_col, labels):
        b = bins[bins[band_col] >= 0]
        g = b.groupby(["period", band_col], observed=True)[["chips", _r]].sum().reset_index()
        g["per_1000"] = 1000 * g[_r] / g["chips"].replace(0, np.nan)
        g["band"] = [labels[i] for i in g[band_col]]
        g["period"] = g["period"].astype(str)
        tot = g.groupby("period")["chips"].transform("sum")
        g["share_of_chips"] = g["chips"] / tot
        return g

    def _heat(g, labels, title):
        return alt.Chart(g).mark_rect().encode(
            x=alt.X("period:N", sort=TAGS, title=None),
            y=alt.Y("band:N", sort=labels, title=title),
            color=alt.Color("per_1000:Q", scale=alt.Scale(scheme="viridis"), title="detections per 1,000 chips"),
            tooltip=["period", "band", alt.Tooltip("per_1000:Q", format=".2f"), alt.Tooltip("chips:Q", format=","),
                     alt.Tooltip("share_of_chips:Q", format=".1%", title="share of the period's chips")],
        ).properties(width="container", height=170)

    _g1 = _rates("n_clear_band", at.NCLEAR_LABELS)
    _g2 = _rates("empty_band", at.EMPTY_LABELS)
    _fig6 = kit.figure(_heat(_g1, at.NCLEAR_LABELS, "n_clear band"), 6, "Detection rate by n_clear band and period.",
                       "Of the scored chips in each n_clear band, how many per 1,000 the chosen rule detects. "
                       "Hover a cell for its number of chips: a band with few chips gives an unsteady rate.",
                       mark="one n_clear band in one period.",
                       colour="detections per 1,000 scored chips in that band.",
                       sample="every scored chip of the selected subregions.")
    _fig7 = kit.figure(_heat(_g2, at.EMPTY_LABELS, "empty_share band"), 7, "Detection rate by how much of the chip was black.",
                       "As Figure 6, grouped by empty_share, the share of the chip's pixels without any clear image.",
                       mark="one empty_share band in one period.",
                       colour="detections per 1,000 scored chips in that band.",
                       sample="every scored chip with empty_share recorded (missing for some chips in periods scored "
                              "before 1 Nov 2026; see the variable list).")
    import marimo as _mo
    _mo.vstack([_fig6, _fig7])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. When a mine is missed, was it cloudy?

    Here only **mines** count: followed chips that the chosen rule detected in at least two
    periods of the chosen frequency. For each of them, every scored period is either *detected*
    or *missed*. If clouds cause misses, the missed periods have worse cloud values than the
    detected periods, and they do so within the same chip.

    We do not decide beforehand what "cloudy" means. Figure 8 shows, for six cloud measures,
    how the detection rate changes along each measure; Figure 9 scores how well each measure
    separates missed from detected periods; Figure 10 shows, for one measure, what a "cloudy"
    cut-off at each value would catch.
    """)
    return


@app.cell(hide_code=True)
def _(A, FREQ, SUBS, TAGS, at, rule):
    _hits = A.table(FREQ, "hits", TAGS)
    _fol = A.static("followed")[["chip_id", "subregion"]]
    _hits = _hits.merge(_fol, on="chip_id", how="left")
    _hits = _hits[_hits["subregion"].isin(SUBS) & _hits["scored"]]
    _n_det = _hits.groupby("chip_id")[rule.value].sum()
    MINES = _n_det.index[_n_det >= 2]
    mine_periods = _hits[_hits["chip_id"].isin(MINES)].assign(detected=lambda d: d[rule.value].astype(bool))
    MEASURES = at.CLOUD_MEASURES
    return MEASURES, MINES, mine_periods


@app.cell(hide_code=True)
def _(MEASURES, MINES, alt, at, kit, mine_periods, mo, pd, rule):
    mo.stop(len(MINES) == 0, mo.md("*No chip was detected in two periods with these settings.*"))
    _rows = []
    for _m, (_label, _dir) in MEASURES.items():
        _g = at.rate_by_bins(mine_periods, _m, "detected", 10)
        _g["measure"] = f"{_m}: {_label}"
        _rows.append(_g)
    _d = pd.concat(_rows, ignore_index=True)
    _base = alt.Chart(_d).encode(x=alt.X("x:Q", title="median value of the measure in the group", scale=alt.Scale(zero=False)))
    _chart = alt.layer(
        _base.mark_area(opacity=0.25, color=kit.DETECTED).encode(y=alt.Y("low:Q", title="share of periods detected",
                                                                           scale=alt.Scale(domain=[0, 1])), y2="high:Q"),
        _base.mark_line(point=True, color=kit.DETECTED).encode(
            y="share:Q", tooltip=[alt.Tooltip("lo:Q", format=".3f", title="from"), alt.Tooltip("hi:Q", format=".3f", title="to"),
                                  alt.Tooltip("n:Q", format=",", title="chip-periods"), alt.Tooltip("share:Q", format=".1%")]),
    ).properties(width=230, height=170).facet(facet=alt.Facet("measure:N", title=None, header=alt.Header(labelLimit=320)),
                                              columns=3).resolve_scale(x="independent")
    kit.figure(_chart, 8, "How often a mine is detected, along each cloud measure.",
               f"Each panel splits the mines' scored periods into ten groups of equal size by one cloud measure and shows "
               f"the share detected in each group. A curve that rises (or falls) steeply marks a measure that matters; "
               f"a flat curve marks one that does not.",
               mark="one tenth of the chip-periods (a decile of the measure); the band is the 95% Wilson interval of the share.",
               x="the measure's median within the group; each panel has its own scale.",
               y="share of the group's chip-periods in which the chip was detected.",
               sample=f"{len(MINES):,} mines (detected in at least two periods by the {at.RULES[rule.value]} rule), "
                      f"{len(mine_periods):,} scored chip-periods.",
               note="groups have equal numbers of chip-periods, so ties in a measure (many chips with empty_share = 0) can merge groups.")
    return


@app.cell(hide_code=True)
def _(MEASURES, MINES, alt, at, kit, mine_periods, mo, pd):
    mo.stop(len(MINES) == 0)
    _rows = []
    for _m, (_label, _dir) in MEASURES.items():
        _x = mine_periods[_m].astype(float) * _dir
        _pooled = at.auc(_x[~mine_periods["detected"]], _x[mine_periods["detected"]])
        _within, _chips, _pairs = at.within_chip_auc(mine_periods, _m, _dir)
        _rows += [{"measure": _m, "comparison": "all mines pooled", "AUC": _pooled, "pairs": None},
                  {"measure": _m, "comparison": "within the same chip", "AUC": _within, "pairs": _pairs, "chips": _chips}]
    auc_table = pd.DataFrame(_rows)
    _order = auc_table[auc_table["comparison"] == "within the same chip"].sort_values("AUC", ascending=False)["measure"].tolist()
    _chart = alt.Chart(auc_table).mark_point(filled=True, size=110).encode(
        y=alt.Y("measure:N", sort=_order, title=None),
        x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.4, 1]), title="AUC (0.5 = no separation, 1 = perfect)"),
        color=alt.Color("comparison:N", scale=alt.Scale(domain=["all mines pooled", "within the same chip"],
                                                        range=["#999999", kit.DETECTED]), title=None),
        shape=alt.Shape("comparison:N", scale=alt.Scale(range=["circle", "diamond"]), title=None),
        tooltip=["measure", "comparison", alt.Tooltip("AUC:Q", format=".3f"), alt.Tooltip("pairs:Q", format=",")],
    ) + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 3], color="black").encode(x="x:Q")
    kit.figure(_chart.properties(width="container", height=260), 9, "Which cloud measure tells missed periods from detected ones?",
               "For each measure, the probability that a missed period had the worse view than a detected period. "
               "Grey compares all missed with all detected periods of all mines; blue compares only periods of the same "
               "chip, so a chip's location cannot explain it. Measures are sorted by the blue bar.",
               mark="one measure and one comparison (grey circle: all mines pooled; blue diamond: within the same chip). The dashed line marks 0.5.",
               x="AUC: share of (missed, detected) pairs in which the missed period had the worse value, ties counted half. "
                 "'Worse' means fewer clear images, more black, lower Cloud Score+, more haze.",
               sample=f"the {len(MINES):,} mines of Figure 8; within-chip pairs use at most 20,000 chips (a fixed random set).")
    return (auc_table,)


@app.cell(hide_code=True)
def _(MEASURES, mo):
    cut_measure = mo.ui.dropdown(options={f"{k}: {v[0]}": k for k, v in MEASURES.items()},
                                 value=f"n_clear: {MEASURES['n_clear'][0]}", label="Measure (Figure 10)")
    cut_measure
    return (cut_measure,)


@app.cell(hide_code=True)
def _(MEASURES, MINES, alt, cut_measure, kit, mine_periods, mo, np, pd):
    mo.stop(len(MINES) == 0)
    _m = cut_measure.value
    _dir = MEASURES[_m][1]
    _x = mine_periods[_m].astype(float)
    _ok = _x.notna()
    _x, _det = _x[_ok], mine_periods.loc[_ok, "detected"]
    _cands = np.unique(np.nanquantile(_x, np.linspace(0.01, 0.99, 60)))
    _rows = []
    for _c in _cands:
        _cloudy = (_x < _c) if _dir > 0 else (_x > _c)
        _rows += [{"cut": _c, "group": "missed periods flagged cloudy", "share": _cloudy[~_det].mean()},
                  {"cut": _c, "group": "detected periods flagged cloudy", "share": _cloudy[_det].mean()}]
    _d = pd.DataFrame(_rows)
    _chart = alt.Chart(_d).mark_line().encode(
        x=alt.X("cut:Q", title=f"cut-off: a period counts as cloudy if {_m} is {'below' if _dir > 0 else 'above'} this value",
                scale=alt.Scale(type="symlog") if _m.startswith("n_clear") else alt.Scale(),
                axis=alt.Axis(values=[0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]) if _m.startswith("n_clear") else alt.Axis()),
        y=alt.Y("share:Q", title="share of periods flagged cloudy", axis=alt.Axis(format="%")),
        color=alt.Color("group:N", scale=alt.Scale(range=[kit.DETECTED, kit.MISSED]), title=None),
        tooltip=[alt.Tooltip("cut:Q", format=".3f"), "group", alt.Tooltip("share:Q", format=".1%")],
    ).properties(width="container", height=260)
    kit.figure(_chart, 10, f"What a 'cloudy' cut-off on {_m} would catch.",
               "For every possible cut-off, the share of the mines' missed periods (orange) and detected periods (blue) "
               "that it would call cloudy. Where the orange line is far above the blue one, the cut-off explains many "
               "misses while flagging few periods in which the mine was found anyway. Where both lines are close, the "
               "cut-off says little about misses.",
               mark="one line per group; one point per candidate cut-off (60 cut-offs at evenly spaced percentiles).",
               x=f"the cut-off value of {_m}" + (" (scale linear near 0, logarithmic above 1)." if _m.startswith("n_clear") else "."),
               y="share of the group's chip-periods flagged cloudy at that cut-off.",
               sample=f"the {len(MINES):,} mines of Figure 8.")
    return


@app.cell(hide_code=True)
def _(MEASURES, MINES, alt, at, kit, mine_periods, mo, pd):
    mo.stop(len(MINES) == 0)
    _rows = []
    for _s, _g in mine_periods.groupby("subregion"):
        for _m, (_label, _dir) in MEASURES.items():
            _a, _chips, _pairs = at.within_chip_auc(_g, _m, _dir, max_chips=5_000)
            _rows.append({"subregion": str(int(_s)), "measure": _m, "AUC": _a, "pairs": _pairs})
    _d = pd.DataFrame(_rows)
    _chart = alt.Chart(_d).mark_rect().encode(
        x=alt.X("subregion:N", title="subregion"), y=alt.Y("measure:N", title=None),
        color=alt.Color("AUC:Q", scale=alt.Scale(scheme="viridis", domain=[0.5, 1]), title="within-chip AUC"),
        tooltip=["subregion", "measure", alt.Tooltip("AUC:Q", format=".3f"), alt.Tooltip("pairs:Q", format=",")],
    ) + alt.Chart(_d).mark_text(fontSize=11).encode(x="subregion:N", y="measure:N", text=alt.Text("AUC:Q", format=".2f"),
                                                     color=alt.value("white"))
    kit.figure(_chart.properties(width="container", height=220), 11, "The within-chip comparison of Figure 9, by subregion.",
               "Whether the same cloud measures explain misses everywhere, or only in some subregions.",
               mark="one subregion and one measure.",
               colour="within-chip AUC (as the blue bars of Figure 9); the number is printed in the cell.",
               sample="the mines of Figure 8, split by the subregion of their tile; at most 5,000 chips per subregion.")
    return


@app.cell(hide_code=True)
def _(A, mo):
    mo.md(f"""
    ---
    <small>Built from `{A.dir}` ({A.log.get('built_utc', '?')}). Every figure's data can be
    reproduced with `panel_pipeline/lib/analysis_tables.py` from a Python prompt.</small>
    """)
    return


if __name__ == "__main__":
    app.run()
