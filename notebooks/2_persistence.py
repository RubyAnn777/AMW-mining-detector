import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="2 Persistence")


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
    FOLLOWED = A.static("followed")
    AMW = A.static("amw")
    return A, AMW, FOLLOWED, PERIODS, RUN


@app.cell(hide_code=True)
def _(A, AMW, PERIODS, RUN, mo):
    _y = sorted(AMW.loc[AMW["frequency"] == "yearly", "year"].unique())
    mo.md(f"""
    # 2. Persistence: when a detected mine is missed, and what AMW's yearly rule makes of it

    A mine is a scar that stays. If the detector finds a chip in one period and not in the next,
    either the mine changed or the detector did not see it. This notebook follows every chip
    that was ever detected through all periods, counts how often and for how long detections
    break off, relates the breaks to clouds, and tests AMW's persistence rule (a mine enters
    AMW's record only when AMW's yearly picture detects it in two consecutive years) against
    what our quarters see.

    **Data.** Run `{RUN.name}`; quarters {', '.join(PERIODS['quarterly']) or 'none'};
    months {PERIODS['monthly'][0] + ' to ' + PERIODS['monthly'][-1] if PERIODS['monthly'] else 'none'}.
    AMW's yearly files: {', '.join(map(str, _y)) or 'none'}. Followed chips: {A.log.get('followed_chips', 0):,}
    (every chip that scored at least 0.2 in some period, is in one of AMW's files, or is a known mine).
    New periods enter every figure when the tables are rebuilt; nothing below names a year.
    """)
    return


@app.cell(hide_code=True)
def _(at, mo):
    frequency = mo.ui.radio(options=["quarterly", "monthly"], value="quarterly", label="Frequency", inline=True)
    subregions = mo.ui.multiselect(options=[str(i) for i in range(1, 7)], value=[], label="Subregions (none = all)")
    rule = mo.ui.dropdown(options={v: k for k, v in at.RULES.items()}, value=at.RULES["det_loose"], label="Detection rule")
    measure = mo.ui.dropdown(options={f"{k}: {v[0]}": k for k, v in at.CLOUD_MEASURES.items()},
                             value=f"n_clear: {at.CLOUD_MEASURES['n_clear'][0]}", label="Cloud measure")
    mo.vstack([mo.hstack([frequency, subregions, rule], justify="start", gap=2), measure,
               mo.md("<small>Frequency, subregions and rule apply to sections 1 and 2; section 3 always uses quarters, "
                     "because AMW's rule is about years and our quarters are what we compare it with.</small>")])
    return frequency, measure, rule, subregions


@app.cell(hide_code=True)
def _(kit):
    kit.glossary(["confidence", "det_raw", "det_loose", "det_stringent", "det_andes", "n_clear", "empty_share",
                  "cs_mean", "subregion", "followed chip", "detected / missed", "gap", "AMW yearly status",
                  "quarterly onset", "known mine"])
    return


@app.cell(hide_code=True)
def _(A, FOLLOWED, PERIODS, at, frequency, rule, subregions):
    FREQ = frequency.value
    TAGS = PERIODS[FREQ]
    SUBS = [int(s) for s in subregions.value] or list(range(1, 7))
    SUB_TEXT = "all subregions" if not subregions.value else "subregion" + ("s " if len(SUBS) > 1 else " ") + ", ".join(map(str, SUBS))
    _chips = FOLLOWED.loc[FOLLOWED["subregion"].isin(SUBS), "chip_id"]
    hits = A.table(FREQ, "hits", TAGS)
    hits = hits[hits["chip_id"].isin(_chips)]
    M = at.detection_matrix(hits, rule.value, TAGS)
    M = M[(M == 1).any(axis=1)]                     # chips detected at least once
    RULE_TEXT = at.RULES[rule.value]
    return FREQ, M, RULE_TEXT, SUBS, SUB_TEXT, TAGS, hits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. From one period to the next
    """)
    return


@app.cell(hide_code=True)
def _(FOLLOWED, M, RULE_TEXT, SUB_TEXT, TAGS, alt, at, kit, mo, pd):
    mo.stop(len(TAGS) < 2, mo.md("*Needs at least two periods.*"))
    _sub = FOLLOWED.set_index("chip_id")["subregion"]
    _rows = []
    for _a, _b in zip(TAGS[:-1], TAGS[1:]):
        for _s, _g in [("all", M)] + [(str(int(k)), v) for k, v in M.groupby(_sub.reindex(M.index).to_numpy())]:
            _base = (_g[_a] == 1) & _g[_b].notna()
            _k = int((_base & (_g[_b] == 1)).sum())
            _n = int(_base.sum())
            _lo, _hi = at.wilson(_k, _n)
            _rows.append({"pair": f"{_a} → {_b}", "to": _b, "subregion": _s, "n": _n, "k": _k,
                          "share": _k / _n if _n else None, "low": float(_lo), "high": float(_hi)})
    retention_steps = pd.DataFrame(_rows)
    _d = retention_steps[retention_steps["subregion"] == "all"]
    _base = alt.Chart(_d).encode(x=alt.X("pair:N", sort=list(_d["pair"]), title=None, axis=alt.Axis(labelAngle=-40)))
    _chart = alt.layer(
        _base.mark_rule(color="#888").encode(y=alt.Y("low:Q", title="share still detected in the next period",
                                                     scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), y2="high:Q"),
        _base.mark_point(filled=True, size=70, color=kit.DETECTED).encode(
            y="share:Q", tooltip=["pair", alt.Tooltip("n:Q", format=",", title="detected in the first period"),
                                  alt.Tooltip("k:Q", format=",", title="also in the next"), alt.Tooltip("share:Q", format=".1%")]),
    ).properties(width="container", height=240)
    _sub_chart = alt.Chart(retention_steps[retention_steps["subregion"] != "all"]).mark_line(point=True).encode(
        x=alt.X("pair:N", sort=list(_d["pair"]), title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("share:Q", title="share still detected", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("subregion:N", title="subregion"),
        tooltip=["pair", "subregion", alt.Tooltip("n:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
    ).properties(width="container", height=220)
    mo.vstack([
        kit.figure(_chart, 1, "Retention from each period to the next.",
                   "Of the chips detected in a period and scored in the next, the share detected again in the next period.",
                   mark="one pair of consecutive periods; the line is the 95% Wilson interval.",
                   y="share detected again in the next period.",
                   sample=f"followed chips of {SUB_TEXT}, {RULE_TEXT} rule."),
        kit.figure(_sub_chart, 2, "Retention by subregion.",
                   "Figure 1 for each subregion separately.",
                   mark="one subregion and one pair of consecutive periods.", y="share detected again in the next period.",
                   colour="subregion of the chip's tile.", sample=f"as Figure 1."),
    ])
    return


@app.cell(hide_code=True)
def _(M, RULE_TEXT, SUB_TEXT, TAGS, alt, at, kit):
    _r = at.retention(M)
    _chart = alt.Chart(_r).mark_rect().encode(
        x=alt.X("to:N", sort=TAGS, title="later (or earlier) period"),
        y=alt.Y("from:N", sort=TAGS, title="period the chip was detected in"),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="share also detected"),
        tooltip=["from", "to", alt.Tooltip("detected in from:Q", format=","), alt.Tooltip("also in to:Q", format=","),
                 alt.Tooltip("share:Q", format=".1%")],
    ).properties(width="container", height=max(200, 22 * len(TAGS)))
    kit.figure(_chart, 3, "Every period against every other period.",
               "For any two periods: of the chips detected in the row's period (and scored in the column's), the share also "
               "detected in the column's period. The cells just right of the diagonal are Figure 1; cells further right show "
               "whether detections survive longer stretches; cells left of the diagonal look backwards.",
               mark="one ordered pair of periods. The diagonal is 100% by definition.",
               colour="share also detected (0 to 100%).",
               sample=f"followed chips of {SUB_TEXT}, {RULE_TEXT} rule.")
    return


@app.cell(hide_code=True)
def _(TAGS, mo):
    pair_a = mo.ui.dropdown(options=TAGS, value=TAGS[0] if TAGS else None, label="Period A")
    pair_b = mo.ui.dropdown(options=TAGS, value=TAGS[1] if len(TAGS) > 1 else None, label="Period B")
    mo.vstack([mo.md("### Two periods side by side"), mo.hstack([pair_a, pair_b], justify="start", gap=2)])
    return pair_a, pair_b


@app.cell(hide_code=True)
def _(M, RULE_TEXT, SUB_TEXT, alt, at, hits, kit, measure, mo, np, pair_a, pair_b, pd):
    mo.stop(not pair_a.value or not pair_b.value or pair_a.value == pair_b.value, mo.md("*Pick two different periods.*"))
    _a, _b = pair_a.value, pair_b.value
    _x, _y = M[_a], M[_b]
    _ok = _x.notna() & _y.notna()
    _grp = pd.Series(np.select([(_x == 1) & (_y == 1), (_x == 1) & (_y == 0), (_x == 0) & (_y == 1)],
                               ["in both", f"only in {_a}", f"only in {_b}"], "neither"), index=M.index)[_ok]
    _grp = _grp[_grp != "neither"]
    _counts = _grp.value_counts().rename_axis("group").reset_index(name="chips")
    _order = ["in both", f"only in {_a}", f"only in {_b}"]
    _colours = [at_c for at_c in (kit.DETECTED, kit.MISSED, kit.YEARLY)]
    _bar = alt.Chart(_counts).mark_bar().encode(
        y=alt.Y("group:N", sort=_order, title=None), x=alt.X("chips:Q", title="chips"),
        color=alt.Color("group:N", scale=alt.Scale(domain=_order, range=_colours), legend=None),
        tooltip=["group", alt.Tooltip("chips:Q", format=",")],
    ).properties(width="container", height=110)
    _m = measure.value
    _v = hits[hits["period"].astype(str).isin([_a, _b])][["chip_id", "period", _m]]
    _v = _v[_v["chip_id"].isin(_grp.index)].assign(group=lambda d: d["chip_id"].map(_grp), period=lambda d: d["period"].astype(str))
    _dens = alt.Chart(_v).transform_density(_m, groupby=["group", "period"], as_=[_m, "density"], extent=[
        float(_v[_m].min()) if len(_v) else 0, float(_v[_m].quantile(0.99)) if len(_v) else 1]).mark_line().encode(
        x=alt.X(f"{_m}:Q", title=f"{_m} in that period"), y=alt.Y("density:Q", title="density"),
        color=alt.Color("group:N", scale=alt.Scale(domain=_order, range=_colours), title=None),
        strokeDash=alt.StrokeDash("period:N", title="period"),
    ).properties(width="container", height=220)
    mo.vstack([
        kit.figure(_bar, 4, f"{_a} against {_b}: which chips each detected.",
                   "Chips detected in both periods, only in A, or only in B, among chips scored in both.",
                   mark="one group; bar length = number of chips.",
                   sample=f"followed chips of {SUB_TEXT}, {RULE_TEXT} rule."),
        kit.figure(_dens, 5, f"How well each group was seen, in {_a} and in {_b}.",
                   f"The distribution of {_m} of the chips of each group of Figure 4, in each of the two periods. "
                   f"If chips found only in A were missed in B because of clouds, their dashed or solid curve for B "
                   f"lies further toward the cloudy end than the curve of chips found in both.",
                   mark="one line per group and period: a smoothed histogram (kernel density) of the measure.",
                   x=f"{_m} of the chip in that period, cut at the 99th percentile.",
                   y="density (the area under each curve is 1, so groups of different size can be compared).",
                   colour="group as in Figure 4.", sample="the chips of Figure 4."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Breaks in detection: how long, and under which sky

    A *gap* is a run of missed periods after a detection. *Closed* gaps end with a new
    detection (D M M D): the mine was there all along, so every M in it is a miss. *Open* gaps
    run to the last period (D M M): the mine may be gone, or still missed. Periods without a
    score are skipped. Any number of misses in a row counts, and the counts grow as periods are added.
    """)
    return


@app.cell(hide_code=True)
def _(M, TAGS, at, hits, measure):
    _m = measure.value
    gaps = at.gap_table(M, {_m: at.value_matrix(hits, _m, TAGS)})
    _dir = at.CLOUD_MEASURES[_m][1]
    _det = at.value_matrix(hits, _m, TAGS).reindex(index=M.index).where(M == 1)
    _worst_when_det = (_det.min(axis=1) if _dir > 0 else _det.max(axis=1)).rename("worst_when_detected")
    gaps = gaps.merge(_worst_when_det, left_on="chip_id", right_index=True, how="left")
    gaps["worse_than_every_detection"] = (gaps[_m] < gaps["worst_when_detected"]) if _dir > 0 else \
        (gaps[_m] > gaps["worst_when_detected"])
    return (gaps,)


@app.cell(hide_code=True)
def _(RULE_TEXT, SUB_TEXT, alt, gaps, kit, mo, np):
    mo.stop(gaps.empty, mo.md("*No gaps with these settings.*"))
    _g = gaps.drop_duplicates("gap").assign(
        kind=lambda d: np.where(d["closed"], "closed (detected again)", "open (not detected since)"),
        length_label=lambda d: np.where(d["length"] >= 6, "6+", d["length"].astype(str)))
    _c = _g.groupby(["length_label", "kind"]).size().reset_index(name="gaps")
    _chart = alt.Chart(_c).mark_bar().encode(
        x=alt.X("length_label:N", sort=["1", "2", "3", "4", "5", "6+"], title="missed periods in a row", axis=alt.Axis(labelAngle=0)),
        xOffset="kind:N", y=alt.Y("gaps:Q", title="number of gaps"),
        color=alt.Color("kind:N", scale=alt.Scale(range=[kit.DETECTED, kit.MISSED_CLOUDY]), title=None),
        tooltip=["length_label", "kind", alt.Tooltip("gaps:Q", format=",")],
    ).properties(width="container", height=240)
    kit.figure(_chart, 6, "How long detections break off.",
               "The number of gaps of each length. A chip can have several gaps.",
               mark="one length and kind of gap; bar height = number of gaps.",
               x="number of consecutive missed periods (scored periods only); 6 or more are pooled.",
               colour="closed: the chip was detected again afterwards; open: not detected again up to the last period.",
               sample=f"followed chips of {SUB_TEXT} detected at least once, {RULE_TEXT} rule.")
    return


@app.cell(hide_code=True)
def _(RULE_TEXT, SUB_TEXT, alt, at, gaps, kit, measure, mo, np, pd):
    mo.stop(gaps.empty)
    _m = measure.value
    _dir = at.CLOUD_MEASURES[_m][1]
    _g = gaps.assign(kind=np.where(gaps["closed"], "closed", "open"),
                     length_label=np.where(gaps["length"] >= 4, "4+", gaps["length"].astype(str)))
    _rows = []
    for (_l, _k), _d in _g.groupby(["length_label", "kind"]):
        _n = int(_d["worse_than_every_detection"].notna().sum())
        _kk = int(_d["worse_than_every_detection"].sum())
        _lo, _hi = at.wilson(_kk, _n)
        _rows.append({"length": _l, "kind": _k, "missed periods": _n, "share": _kk / _n if _n else None,
                      "low": float(_lo), "high": float(_hi)})
    _d = pd.DataFrame(_rows)
    _base = alt.Chart(_d).encode(x=alt.X("length:N", sort=["1", "2", "3", "4+"], title="length of the gap the missed period is in",
                                         axis=alt.Axis(labelAngle=0)),
                                 xOffset="kind:N")
    _chart = alt.layer(
        _base.mark_bar(opacity=0.85).encode(y=alt.Y("share:Q", title="share worse than every detected period", axis=alt.Axis(format="%"),
                                                    scale=alt.Scale(domain=[0, 1])),
                                            color=alt.Color("kind:N", scale=alt.Scale(range=[kit.DETECTED, kit.MISSED_CLOUDY]), title=None)),
        _base.mark_rule(color="black").encode(y="low:Q", y2="high:Q"),
    ).encode(tooltip=["length", "kind", alt.Tooltip("missed periods:Q", format=","), alt.Tooltip("share:Q", format=".1%")]
             ).properties(width="container", height=240)
    kit.figure(_chart, 7, f"Missed periods that were cloudier than any period the mine was found in.",
               f"For each missed period, whether its {_m} was worse than in every period in which the same chip was detected. "
               f"This needs no cloud threshold: each mine sets its own. A high share means the misses happen when the view "
               f"is worse than the mine has ever been found under; a low share means the mine was missed under a sky it is "
               f"usually found under.",
               mark="one gap length and kind; the black line is the 95% Wilson interval.",
               x="length of the gap the missed period belongs to (4 or more pooled).",
               y=f"share of missed periods with {_m} {'below the lowest' if _dir > 0 else 'above the highest'} value among the chip's detected periods.",
               colour="closed or open gap, as in Figure 6.",
               sample=f"every missed period of Figure 6's gaps with {_m} recorded.",
               note="a chip detected only once has one detected period to compare with, which makes 'worse than every detection' easier to reach.")
    return


@app.cell(hide_code=True)
def _(M, RULE_TEXT, SUB_TEXT, TAGS, alt, at, kit, mo, pd):
    mo.stop(M.empty)
    _p = at.pattern_strings(M).value_counts().head(25)
    _rows = []
    for _rank, (_pat, _n) in enumerate(_p.items()):
        for _j, _ch in enumerate(_pat):
            _rows.append({"pattern": _pat, "rank": _rank, "period": TAGS[_j],
                          "state": {"D": "detected", "M": "missed", "·": "not scored"}[_ch], "chips": int(_n)})
    _d = pd.DataFrame(_rows)
    _order = list(_p.index)
    _grid = alt.Chart(_d).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("period:N", sort=TAGS, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("pattern:N", sort=_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("state:N", scale=alt.Scale(domain=["detected", "missed", "not scored"],
                                                   range=[kit.DETECTED, kit.MISSED, kit.NOT_SCORED]), title=None),
        tooltip=["period", "state", alt.Tooltip("chips:Q", format=",")],
    ).properties(width=min(900, 40 * len(TAGS)), height=18 * len(_order))
    _bars = alt.Chart(_d.drop_duplicates("pattern")).mark_bar(color="#555").encode(
        y=alt.Y("pattern:N", sort=_order, title=None, axis=None), x=alt.X("chips:Q", title="chips"),
        tooltip=[alt.Tooltip("chips:Q", format=",")]).properties(width=160, height=18 * len(_order))
    kit.figure(alt.hconcat(_grid, _bars, spacing=4), 8, "The 25 most common detection histories.",
               "Each row is one sequence of detected and missed periods; the bar on the right counts the chips that have "
               "exactly this sequence.",
               mark="one row per sequence; one square per period.",
               colour="detected, missed (scored but not detected) or not scored.",
               sample=f"followed chips of {SUB_TEXT} detected at least once, {RULE_TEXT} rule. "
                      f"{len(M):,} chips, {M.shape[1]} periods.")
    return


@app.cell(hide_code=True)
def _(M, mo):
    _default = str(M.index[0]) if len(M) else ""
    chip_box = mo.ui.text(value=_default, label="Follow one chip (chip_id)", full_width=False)
    mo.vstack([mo.md("### One chip through time"), chip_box])
    return (chip_box,)


@app.cell(hide_code=True)
def _(A, FOLLOWED, PERIODS, alt, at, chip_box, kit, mo, pd):
    try:
        _cid = int(chip_box.value.strip())
    except ValueError:
        _cid = None
    mo.stop(_cid is None, mo.md("*Type a chip_id.*"))
    _rows = []
    for _f, _tags in PERIODS.items():
        _h = A.table(_f, "hits", _tags)
        _h = _h[_h["chip_id"] == _cid]
        if len(_h):
            _rows.append(_h.assign(frequency=_f))
    mo.stop(not _rows, mo.md(f"*Chip {_cid} is not a followed chip.*"))
    _d = pd.concat(_rows, ignore_index=True)
    _d["period"] = _d["period"].astype(str)
    _d["date"] = [at.period_middle(t) for t in _d["period"]]
    _info = FOLLOWED[FOLLOWED["chip_id"] == _cid].iloc[0]
    _long = pd.concat([_d.assign(series="confidence", value=_d["confidence"]),
                       _d.assign(series="n_clear", value=_d["n_clear"])])
    _chart = alt.Chart(_long).mark_line(point=True).encode(
        x=alt.X("date:T", title="middle of the period", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("frequency:N", scale=alt.Scale(domain=["monthly", "quarterly"], range=[kit.MONTHLY, kit.QUARTERLY]), title=None),
        tooltip=["period", alt.Tooltip("value:Q", format=".3f"), "det_loose", "det_stringent",
                 alt.Tooltip("empty_share:Q", format=".2f")],
    ).properties(width="container", height=170).facet(row=alt.Row("series:N", title=None)).resolve_scale(y="independent")
    _where = f"lon {_info['lon']:.5f}, lat {_info['lat']:.5f}"
    _link = f"https://www.google.com/maps/@{_info['lat']:.5f},{_info['lon']:.5f},16z/data=!3m1!1e3"
    kit.figure(_chart, 9, f"Chip {_cid} through time.",
               f"The chip's score and its number of clear images in every month and quarter. Located at {_where} "
               f"(subregion {int(_info['subregion'])}, {_info['country']}; [satellite view]({_link})); "
               f"{'a known mine since ' + str(int(_info['known_since'])) if _info['known'] else 'not a known mine'}.",
               mark="one period, at its middle.", y="top: confidence (0 to 1); bottom: n_clear (clear images per pixel).",
               colour="frequency.", sample="this chip only.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. AMW's yearly persistence rule, tested against our quarters

    **The rule** (AMW's recipe A, `core/persistence.py`). A chip enters AMW's confirmed record
    when AMW's yearly picture (the median of a whole year's clear images) detects it in a year
    *and* in the next year. A chip detected only in AMW's latest year stays *provisional*
    (shown only if it also passes the stricter stringent rule) until the next year's picture
    exists. Example: a mine that appears in October 2025 is mostly forest in the 2025 median,
    so the 2025 picture may miss it; if 2026 detects it, it is provisional in 2026 and can only
    be confirmed by 2027.

    **The test.** Our quarters see such a mine from 2025Q4 on. The figures below measure how
    often a yearly picture misses mines that our quarters of the same year detect, whether
    that depends on when in the year the mine first appeared, and how long a mine our quarters
    see repeatedly stays out of AMW's confirmed record.
    """)
    return


@app.cell(hide_code=True)
def _(A, AMW, FOLLOWED, PERIODS, at, rule, subregions):
    _subs = [int(s) for s in subregions.value] or list(range(1, 7))
    _chips = FOLLOWED.loc[FOLLOWED["subregion"].isin(_subs), "chip_id"]
    QT = PERIODS["quarterly"]
    _hq = A.table("quarterly", "hits", QT)
    _hq = _hq[_hq["chip_id"].isin(_chips)]
    MQ = at.detection_matrix(_hq, rule.value, QT)
    MQ = MQ[(MQ == 1).any(axis=1)]
    NCQ = at.value_matrix(_hq, "n_clear", QT).reindex(MQ.index)
    YEARLY = AMW[AMW["frequency"] == "yearly"]
    YEARS_AMW = sorted(YEARLY["year"].unique().tolist())
    STATUS = at.yearly_status(AMW)
    return MQ, NCQ, QT, STATUS, YEARLY, YEARS_AMW


@app.cell(hide_code=True)
def _(MQ, NCQ, QT, RULE_TEXT, YEARLY, YEARS_AMW, alt, at, kit, mo, np, pd):
    # Per year with both our four quarters and an AMW yearly file: did the yearly picture catch
    # the chips our quarters detected, by the quarter they first appeared in that year?
    _years = sorted({int(q[:4]) for q in QT if sum(t.startswith(q[:4]) for t in QT) == 4} & set(YEARS_AMW))
    mo.stop(not _years, mo.md("*Section 3 needs a year with all four of our quarters and AMW's yearly file of that year.*"))
    _rows, _mech = [], []
    for _y in _years:
        _qs = [q for q in QT if q.startswith(str(_y))]
        _m = MQ[_qs]
        _det = (_m == 1)
        _any = _det.any(axis=1)
        _first = _det.idxmax(axis=1).where(_any)
        _earlier = MQ[[q for q in QT if q < _qs[0]]].eq(1).any(axis=1) if QT.index(_qs[0]) > 0 else pd.Series(False, index=MQ.index)
        _yearly = set(YEARLY.loc[(YEARLY["year"] == _y) & YEARLY["det_loose"], "chip_id"])
        _in = pd.Series(MQ.index.isin(list(_yearly)), index=MQ.index)
        # a chip counts as new in a quarter if our quarters scored it and did not detect it in the quarter before;
        # chips first detected in the very first quarter we have cannot be told apart and are left out
        _prev = {q: (QT[QT.index(q) - 1] if QT.index(q) > 0 else None) for q in _qs}
        _missed_before = pd.Series({c: (_prev[_first[c]] is not None and MQ.loc[c, _prev[_first[c]]] == 0)
                                    if _any[c] else False for c in MQ.index})
        for _q in _qs:
            for _new, _label in ((True, "new: not detected in the quarter before"), (False, "detected in an earlier year too")):
                _sel = (_first == _q) & ((~_earlier & _missed_before) if _new else _earlier)
                _n, _k = int(_sel.sum()), int((_sel & _in).sum())
                _lo, _hi = at.wilson(_k, _n)
                _rows.append({"year": str(_y), "quarter": _q[-2:], "group": _label, "chips": _n,
                              "share": _k / _n if _n else None, "low": float(_lo), "high": float(_hi)})
        # the mechanism: what share of the year's clear images came after the chip first appeared
        _nc = NCQ[_qs]
        _after = pd.Series([_nc.loc[c, _qs[_qs.index(_first[c]):]].sum() / _nc.loc[c].sum() if _any[c] and _nc.loc[c].sum() > 0 else np.nan
                            for c in MQ.index], index=MQ.index)
        _mech.append(pd.DataFrame({"year": str(_y), "after": _after, "in_yearly": _in})[_any & ~_earlier & _missed_before])
    caught = pd.DataFrame(_rows)
    MECH = pd.concat(_mech, ignore_index=True)
    _base = alt.Chart(caught).encode(x=alt.X("quarter:N", title="quarter of the year in which our quarters first detected the chip",
                                             axis=alt.Axis(labelAngle=0)),
                                     xOffset="group:N")
    _chart = alt.layer(
        _base.mark_bar(opacity=0.85).encode(y=alt.Y("share:Q", title="share also detected by AMW's yearly picture",
                                                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                            color=alt.Color("group:N", scale=alt.Scale(range=[kit.AMW, "#999999"]), title=None)),
        _base.mark_rule(color="black").encode(y="low:Q", y2="high:Q"),
    ).encode(tooltip=["year", "quarter", "group", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".1%")]
             ).properties(width=360, height=240).facet(column=alt.Column("year:N", title=None))
    kit.figure(_chart, 10, "Does AMW's yearly picture catch what our quarters detect?",
               "Chips our quarters detected in a year, grouped by the first quarter of that year in which they were detected, "
               "and the share of them that AMW's yearly picture of the same year also detected (loose rule). If mines that "
               "appear late in the year are missed by the yearly median, the orange bars fall from Q1 to Q4.",
               mark="one quarter and group; the black line is the 95% Wilson interval.",
               x="the first quarter of the year with a detection by our quarters.",
               y="share of those chips in AMW's yearly file of the year at the loose rule.",
               colour="orange: new chips, scored and not detected in the quarter before their first detection, and never "
                      "detected in an earlier year; grey: chips our quarters had already detected in an earlier year. Chips first "
                      "detected in the first quarter we have scored cannot be called new or old and are left out.",
               sample=f"followed chips detected by our quarters ({RULE_TEXT} rule) in years where all four quarters and AMW's "
                      f"yearly file exist: {', '.join(map(str, _years))}.",
               note="'new' is relative to the first quarter we have scored; when 2024 and earlier are added, fewer chips will count as new.")
    return (MECH,)


@app.cell(hide_code=True)
def _(MECH, alt, at, kit, mo):
    mo.stop(MECH.empty or MECH["after"].notna().sum() < 20, mo.md("*Too few new chips for Figure 11.*"))
    _g = at.rate_by_bins(MECH.dropna(subset=["after"]).assign(in_yearly=lambda d: d["in_yearly"].astype(bool)),
                         "after", "in_yearly", 6)
    _base = alt.Chart(_g).encode(x=alt.X("x:Q", title="share of the year's clear images taken from the quarter the chip first "
                                                      "appeared in onwards", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
    _chart = alt.layer(
        _base.mark_area(opacity=0.25, color=kit.AMW).encode(y=alt.Y("low:Q", title="share caught by AMW's yearly picture",
                                                                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), y2="high:Q"),
        _base.mark_line(point=True, color=kit.AMW).encode(y="share:Q", tooltip=[alt.Tooltip("n:Q", format=","), alt.Tooltip("share:Q", format=".1%")]),
        alt.Chart({"values": [{"x": 0.5}]}).mark_rule(strokeDash=[4, 3]).encode(x="x:Q"),
    ).properties(width="container", height=240)
    kit.figure(_chart, 11, "Why a yearly median misses new mines.",
               "A yearly picture is the median of the year's clear images, pixel by pixel. If most clear images of a pixel were "
               "taken before the mine appeared, the median shows the forest. The horizontal axis is the share of the chip's "
               "clear images in that year that were taken from the quarter of first detection onwards (estimated from our "
               "quarterly n_clear); the dashed line marks half. Above half, the median is mostly made of post-mine images.",
               mark="one sixth of the chips (groups of equal size; fewer groups where many chips share a value); "
                    "the band is the 95% Wilson interval.",
               x="share of the year's clear images from the first-detection quarter onwards (sum of n_clear of those quarters "
                 "divided by the year's sum).",
               y="share of the group's chips that AMW's yearly picture detected (loose rule).",
               sample="the 'new' chips of Figure 10 (orange bars).",
               note="n_clear is an average over the chip's pixels; the median is taken per pixel, so this is an approximation.")
    return


@app.cell(hide_code=True)
def _(MQ, QT, RULE_TEXT, STATUS, YEARS_AMW, alt, at, kit, mo, pd):
    _onset = at.quarterly_onset(MQ, window=4).dropna()
    mo.stop(_onset.empty, mo.md("*No chip was detected twice within four quarters.*"))
    _st = STATUS.set_index("chip_id")["status"]
    _d = pd.DataFrame({"quarterly onset": _onset, "AMW status": _st.reindex(_onset.index).fillna("in no AMW yearly file")})
    _last = max(YEARS_AMW) if YEARS_AMW else None
    _order = ["confirmed", f"provisional ({_last} only)", f"{_last} only, below the provisional rule", "never confirmed",
              "in no AMW yearly file"]
    _c = _d.groupby(["quarterly onset", "AMW status"]).size().reset_index(name="chips")
    _c["share"] = _c["chips"] / _c.groupby("quarterly onset")["chips"].transform("sum")
    _chart = alt.Chart(_c).mark_bar().encode(
        x=alt.X("quarterly onset:N", sort=QT, title="quarterly onset (first of two detections within four quarters)"),
        y=alt.Y("share:Q", title="share of chips", axis=alt.Axis(format="%"), stack="normalize"),
        color=alt.Color("AMW status:N", sort=_order, scale=alt.Scale(domain=_order,
                        range=[kit.DETECTED, "#56B4E9", "#E69F00", "#CC79A7", "#BBBBBB"]), title="status under AMW's yearly rule"),
        order=alt.Order("order:Q"),
        tooltip=["quarterly onset", "AMW status", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
    ).transform_calculate(order=f"indexof({_order}, datum['AMW status'])").properties(width="container", height=260)
    kit.figure(_chart, 12, "Mines our quarters see twice: where they stand in AMW's record.",
               "Every chip that our quarters detected twice within four quarters (a quarterly version of AMW's one-year window), "
               "by the quarter of the first of those detections, and its status under AMW's yearly rule today.",
               mark="one quarterly-onset quarter; the bar is split into shares of its chips (bars sum to 100%).",
               x="the chip's quarterly onset.",
               y="share of the chips with that onset.",
               colour=f"confirmed: AMW's yearly picture detected it in two consecutive years; provisional: detected only in "
                      f"{_last} at the stringent rule, waiting for {(_last or 0) + 1}; '{_last} only, below the provisional rule': "
                      f"detected in {_last} at the loose rule only, so not shown; never confirmed: detected in earlier years "
                      f"but never two in a row; in no AMW yearly file: AMW's yearly pictures never detected it.",
               sample=f"followed chips with a quarterly onset ({RULE_TEXT} rule for our quarters; AMW's statuses use AMW's loose "
                      f"and stringent rules on its yearly files {YEARS_AMW[0] if YEARS_AMW else ''} to {_last}).")
    return


@app.cell(hide_code=True)
def _(MQ, QT, STATUS, YEARS_AMW, alt, at, kit, mo, np, pd):
    _onset = at.quarterly_onset(MQ, window=4).dropna()
    mo.stop(_onset.empty or not YEARS_AMW)
    _last = max(YEARS_AMW)
    _st = STATUS.set_index("chip_id")
    _rows = []
    for _c, _q in _onset.items():
        _qy = int(_q[:4])
        _qi = (_qy * 12 + (int(_q[-1]) - 1) * 3 + 3)                 # end of the onset quarter, in months
        if _c in _st.index and _st.loc[_c, "status"] == "confirmed":
            _year = int(_st.loc[_c, "confirmed_onset"]) + 1
            _kind = "confirmed (year of confirmation)"
        else:
            _years = [int(y) for y in str(_st.loc[_c, "years"]).split()] if _c in _st.index else []
            # earliest possible: detected in the onset year's picture -> confirmed with the next year; otherwise one year later
            _year = (_qy + 1) if _qy in _years else (_qy + 2)
            _year = max(_year, _last + 1)
            _kind = "not confirmed yet (earliest possible year)"
        _rows.append({"chip_id": _c, "onset": _q, "kind": _kind, "delay_quarters": ((_year + 1) * 12 - _qi) / 3})
    _d = pd.DataFrame(_rows)
    _d = _d[_d["delay_quarters"] >= 0]
    _chart = alt.Chart(_d).mark_bar().encode(
        x=alt.X("delay_quarters:O", axis=alt.Axis(labelAngle=0), title="quarters from the end of the quarterly onset to the end of the year AMW's rule confirms (or can first confirm) the chip"),
        y=alt.Y("count():Q", title="chips"),
        color=alt.Color("kind:N", scale=alt.Scale(range=[kit.DETECTED, "#BBBBBB"]), title=None),
        tooltip=["delay_quarters", "kind", alt.Tooltip("count():Q", format=",")],
    ).properties(width="container", height=240)
    kit.figure(_chart, 13, "How long a mine our quarters see stays out of AMW's confirmed record.",
               "For every chip with a quarterly onset: the time from the end of its onset quarter to the end of the year in "
               "which AMW's yearly rule confirmed it (blue) or, for chips not confirmed yet, the earliest year it could be "
               "confirmed (grey): the year after next if the onset year's picture missed it, the next year if it caught it, "
               "and never earlier than the year after AMW's latest file. Negative delays (AMW confirmed the chip before our "
               "quarters saw it twice) are left out.",
               mark="one delay; bar height = number of chips.",
               x="delay in quarters. A chip with onset 2025Q4 that the 2025 picture missed can be confirmed at the end of 2027 at the "
                 "earliest: 8 quarters.",
               colour="confirmed already, or not yet (earliest possible year assumed).",
               sample="the chips of Figure 12.",
               note="grey bars are lower bounds: they assume AMW's next yearly pictures will detect the chip.")
    return


@app.cell(hide_code=True)
def _(A, mo):
    mo.md(f"""
    ---
    <small>Built from `{A.dir}` ({A.log.get('built_utc', '?')}). The measures are functions in
    `panel_pipeline/lib/analysis_tables.py` (detection_matrix, retention, gap_table, yearly_status, quarterly_onset).</small>
    """)
    return


if __name__ == "__main__":
    app.run()
