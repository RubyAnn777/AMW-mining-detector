import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="4 Months, quarters and years")


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
    AMW_ALL = A.static("amw")
    COMPLETE_Q = [q for q in PERIODS["quarterly"] if all(m in PERIODS["monthly"] for m in at.months_of(q))]
    return AMW_ALL, A, COMPLETE_Q, FOLLOWED, PERIODS, RUN


@app.cell(hide_code=True)
def _(AMW_ALL, COMPLETE_Q, PERIODS, RUN, mo):
    _amw_y = sorted(AMW_ALL.loc[AMW_ALL["frequency"] == "yearly", "period"].unique())
    _amw_q = sorted(AMW_ALL.loc[AMW_ALL["frequency"] == "quarterly", "period"].unique())
    def _span(tags):
        return f"{tags[0]} to {tags[-1]} ({len(tags)})" if tags else "none"
    mo.md(f"""
    # 4. Months, quarters and years: what each picture finds

    The same ground is scored from pictures of different lengths: one per month, one per
    quarter (ours), and one per quarter and one per year (AMW's). A longer period gives more
    chances of a clear view, a shorter one shows change sooner. This notebook counts what each
    picture finds, how the detections of the short periods add up to those of the long ones,
    and, chip by chip, what a month missed that its quarter found (and the reverse).

    **Data.** Run `{RUN.name}`.
    Our quarters: {_span(PERIODS['quarterly'])}. Our months: {_span(PERIODS['monthly'])}.
    Quarters with all three months assembled: {", ".join(COMPLETE_Q) or "none"}.
    AMW yearly files: {_span(_amw_y)}. AMW quarterly files: {_span(_amw_q)}.

    Every detected chip is in the prepared tables (all chips that ever score 0.2 or more are
    followed), so all counts here are complete, not sampled.
    """)
    return


@app.cell(hide_code=True)
def _(at, mo):
    subregions = mo.ui.multiselect(options=[str(i) for i in range(1, 7)], value=[], label="Subregions (none = all)")
    rule = mo.ui.dropdown(options={v: k for k, v in at.RULES.items() if k != "det_andes"},
                          value=at.RULES["det_loose"], label="Detection rule")
    mo.vstack([mo.hstack([subregions, rule], justify="start", gap=2),
               mo.md("<small>The rule applies to our scores and to AMW's scores alike. AMW's files list only chips "
                     "scoring 0.4 or more, so under the raw rule an AMW detection is any listed chip.</small>")])
    return rule, subregions


@app.cell(hide_code=True)
def _(kit):
    kit.glossary(["confidence", "amw_confidence", "det_raw", "det_loose", "det_stringent", "source", "unique chips",
                  "quarter only / month only / both", "months that caught it", "n_clear", "n_clear band",
                  "empty_share", "subregion"])
    return


@app.cell(hide_code=True)
def _(AMW_ALL, A, FOLLOWED, PERIODS, at, pd, rule, subregions):
    SUBS = [int(s) for s in subregions.value] or list(range(1, 7))
    SUB_TEXT = "all subregions" if not subregions.value else "subregion" + ("s " if len(SUBS) > 1 else " ") + ", ".join(map(str, SUBS))
    RULE = rule.value
    _mine = FOLLOWED.loc[FOLLOWED["subregion"].isin(SUBS), ["chip_id", "subregion", "lon", "lat", "country"]]
    _keep = set(_mine["chip_id"])

    def _hits(freq):
        h = A.table(freq, "hits", PERIODS[freq])
        if h.empty:
            return h
        h = h[h["chip_id"].isin(_keep)].copy()
        h["period"] = h["period"].astype(str)
        h["detected"] = h[RULE].fillna(False).astype(bool) & h["scored"]
        return h

    QH, MH = _hits("quarterly"), _hits("monthly")
    _amw = AMW_ALL[AMW_ALL["chip_id"].isin(_keep)].copy()
    _amw["detected"] = True if RULE == "det_raw" else _amw[RULE].astype(bool)
    _parts = [
        MH.loc[MH["detected"], ["chip_id", "period"]].assign(source="our months"),
        QH.loc[QH["detected"], ["chip_id", "period"]].assign(source="our quarters"),
        _amw.loc[_amw["detected"] & (_amw["frequency"] == "quarterly"), ["chip_id", "period"]].assign(source="AMW quarters"),
        _amw.loc[_amw["detected"] & (_amw["frequency"] == "yearly"), ["chip_id", "period"]].assign(source="AMW yearly"),
    ]
    DET = pd.concat(_parts, ignore_index=True).merge(_mine[["chip_id", "subregion"]], on="chip_id")
    _spans = {t: at.period_span(t) for t in DET["period"].unique()}
    DET["start"] = DET["period"].map(lambda t: _spans[t][0])
    DET["end"] = DET["period"].map(lambda t: _spans[t][1])
    DET["year"] = DET["start"].dt.year
    # every period each source has (a period with no detection still counts as 0)
    PERIODS_OF = {"our months": PERIODS["monthly"], "our quarters": PERIODS["quarterly"],
                  "AMW quarters": sorted(AMW_ALL.loc[AMW_ALL["frequency"] == "quarterly", "period"].unique()),
                  "AMW yearly": sorted(AMW_ALL.loc[AMW_ALL["frequency"] == "yearly", "period"].unique())}
    SOURCES = ["our months", "our quarters", "AMW quarters", "AMW yearly"]
    CHIPS = _mine
    return CHIPS, DET, MH, PERIODS_OF, QH, RULE, SOURCES, SUBS, SUB_TEXT


@app.cell(hide_code=True)
def _(kit):
    SOURCE_COLOURS = {"our months": "#56B4E9", "our quarters": kit.DETECTED, "AMW quarters": "#E69F00", "AMW yearly": kit.AMW}
    return (SOURCE_COLOURS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How many chips each picture detects
    """)
    return


@app.cell(hide_code=True)
def _(DET, PERIODS_OF, RULE, SOURCES, SOURCE_COLOURS, SUBS, SUB_TEXT, alt, at, kit, mo, pd):
    def _counts(by):
        rows = []
        for src in SOURCES:
            for tag in PERIODS_OF[src]:
                s, e = at.period_span(tag)
                for key in (SUBS if by else [None]):
                    sel = (DET["source"] == src) & (DET["period"] == tag)
                    if key is not None:
                        sel &= DET["subregion"] == key
                    rows.append({"source": src, "period": tag, "start": s, "end": e, "subregion": key,
                                 "chips": int(DET.loc[sel, "chip_id"].nunique())})
        d = pd.DataFrame(rows)
        return d[d["start"] >= pd.Timestamp("2024-01-01")] if (d["start"] >= pd.Timestamp("2024-01-01")).any() else d

    def _chart(d, width, height):
        enc = dict(x=alt.X("start:T", title=None), x2="end:T", y=alt.Y("chips:Q", title="detected chips"),
                   color=alt.Color("source:N", sort=SOURCES, scale=alt.Scale(domain=SOURCES, range=[SOURCE_COLOURS[s] for s in SOURCES]),
                                   title=None),
                   tooltip=["source", "period", alt.Tooltip("chips:Q", format=",")])
        return alt.Chart(d).mark_rule(strokeWidth=4).encode(**enc).properties(width=width, height=height)

    _all = _counts(False)
    _fig1 = kit.figure(_chart(_all, "container", 280), 1, "Detections per period, by source.",
                       "Each horizontal stroke is one period of one source: it spans the dates the picture covers, at the "
                       "height of the number of chips detected in it. A month, a quarter and a year of the same place are "
                       "drawn over the same dates, so strokes of different lengths can be compared directly.",
                       mark="one period of one source.", x="the dates the period covers.",
                       y=f"number of chips detected in that period ({at.RULES[RULE]}).",
                       colour="source (our months, our quarters, AMW quarters, AMW yearly).",
                       sample=f"every chip of {SUB_TEXT}; periods from 2024 on (AMW's earlier years serve as known mines).")
    _items = [_fig1]
    if len(SUBS) > 1:
        _sub = _counts(True)
        _sub = _sub[_sub["subregion"].isin(DET["subregion"].unique())]
        _c = _chart(_sub, 300, 150).facet(facet=alt.Facet("subregion:N", title="subregion"), columns=3).resolve_scale(y="independent")
        _items.append(kit.figure(_c, 2, "The same, subregion by subregion.",
                                 "Figure 1 split by subregion. Each panel has its own vertical scale, so compare the shape "
                                 "of the strokes within a panel, not their heights across panels.",
                                 mark="one period of one source in one subregion.", x="dates.", y="detected chips.",
                                 colour="source.", sample="every chip of the subregion."))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Months adding up to their quarter

    For each quarter whose three months are assembled: how many different chips the months find
    together, compared with the quarter's own picture. If the months found everything the
    quarter found, each month line would reach its quarter's square.
    """)
    return


@app.cell(hide_code=True)
def _(COMPLETE_Q, DET, at, mo, pd):
    mo.stop(not COMPLETE_Q, mo.callout(mo.md("No quarter has all three of its months assembled yet."), kind="info"))
    _sets = {(s, p): set(g["chip_id"]) for (s, p), g in DET.groupby(["source", "period"])}
    def _get(src, tag):
        return _sets.get((src, tag), set())
    _rows, _parts, _caught = [], [], []
    for _q in COMPLETE_Q:
        _ms = at.months_of(_q)
        _qs = _get("our quarters", _q)
        _acc = set()
        for _i, _m in enumerate(_ms):
            _acc |= _get("our months", _m)
            _rows.append({"quarter": _q, "step": ["month 1", "months 1 and 2", "months 1 to 3"][_i], "chips": len(_acc)})
        _rows.append({"quarter": _q, "step": "the quarter", "chips": len(_qs)})
        _rows.append({"quarter": _q, "step": "quarter or any month", "chips": len(_qs | _acc)})
        _parts += [{"quarter": _q, "group": "quarter only", "chips": len(_qs - _acc)},
                   {"quarter": _q, "group": "quarter and at least one month", "chips": len(_qs & _acc)},
                   {"quarter": _q, "group": "months only", "chips": len(_acc - _qs)}]
        for _c in _qs | _acc:
            _caught.append({"quarter": _q, "in quarter": _c in _qs, "months": sum(_c in _get("our months", _m) for _m in _ms)})
    CUM_Q = pd.DataFrame(_rows)
    PARTS_Q = pd.DataFrame(_parts)
    CAUGHT = pd.DataFrame(_caught)
    return CAUGHT, CUM_Q, PARTS_Q


@app.cell(hide_code=True)
def _(CAUGHT, CUM_Q, PARTS_Q, SUB_TEXT, alt, at, kit, mo, pd):
    _steps = ["month 1", "months 1 and 2", "months 1 to 3", "the quarter", "quarter or any month"]
    _enc = dict(x=alt.X("step:N", sort=_steps, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("chips:Q", title="unique detected chips"),
                color=alt.Color("quarter:N", scale=alt.Scale(scheme="viridis"), title="quarter"))
    _c3 = alt.layer(
        alt.Chart(CUM_Q[CUM_Q["step"].str.startswith("month")]).mark_line().encode(**_enc),
        alt.Chart(CUM_Q.assign(kind=[("months" if st.startswith("month") else "quarter" if st == "the quarter"
                                      else "quarter or months") for st in CUM_Q["step"]])).mark_point(filled=True, size=90).encode(
            **_enc, shape=alt.Shape("kind:N", scale=alt.Scale(domain=["months", "quarter", "quarter or months"],
                                                               range=["circle", "square", "diamond"]), title=None),
            tooltip=["quarter", "step", alt.Tooltip("chips:Q", format=",")]),
    ).properties(width="container", height=260)
    _groups = ["quarter only", "quarter and at least one month", "months only"]
    _c4 = alt.Chart(PARTS_Q).mark_bar().encode(
        y=alt.Y("quarter:N", title=None), x=alt.X("chips:Q", title="unique detected chips"),
        color=alt.Color("group:N", sort=_groups, scale=alt.Scale(domain=_groups, range=[kit.DETECTED, "#7f7f7f", "#56B4E9"]),
                        title=None),
        order=alt.Order("order:Q"),
        tooltip=["quarter", "group", alt.Tooltip("chips:Q", format=",")],
    ).transform_calculate(order=f"indexof({_groups}, datum.group)").properties(width="container", height=32 * PARTS_Q["quarter"].nunique())
    _k = CAUGHT[CAUGHT["in quarter"]].groupby(["quarter", "months"]).size().reset_index(name="chips")
    _k["share"] = _k["chips"] / _k.groupby("quarter")["chips"].transform("sum")
    _k["months that caught it"] = _k["months"].astype(str) + " of 3"
    _c5 = alt.Chart(_k).mark_bar().encode(
        y=alt.Y("quarter:N", title=None), x=alt.X("share:Q", title="share of the quarter's detected chips", axis=alt.Axis(format="%")),
        color=alt.Color("months that caught it:N", sort=["0 of 3", "1 of 3", "2 of 3", "3 of 3"],
                        scale=alt.Scale(domain=["0 of 3", "1 of 3", "2 of 3", "3 of 3"], range=["#E69F00", "#c6dbef", "#6baed6", "#08519c"]),
                        title="months of the quarter that also detected the chip"),
        order=alt.Order("months:Q"),
        tooltip=["quarter", "months that caught it", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
    ).properties(width="container", height=32 * _k["quarter"].nunique())
    mo.vstack([
        kit.figure(_c3, 3, "Unique chips as months are added, against the quarter.",
                   "For each quarter, the number of different chips detected in its first month, in its first two months, "
                   "in all three, in the quarter's own picture, and in the quarter or any of its months. The line joins only "
                   "the month steps; if it ends below the square, the months together missed chips the quarter found.",
                   mark="one step of one quarter; circle = months, square = the quarter, diamond = quarter or any month.", x="which pictures are combined (left to right, more pictures).",
                   y="unique chips detected by at least one of the combined pictures.", colour="quarter.",
                   sample=f"every chip of {SUB_TEXT}; quarters whose three months are assembled."),
        kit.figure(_c4, 4, "Who finds what: the quarter, its months, or both.",
                   "Every chip detected in a quarter or in one of its months, split by whether the quarter's picture "
                   "found it, the months found it, or both did.",
                   mark="one group of one quarter; bar length = number of chips.", x="unique detected chips.",
                   colour="dark blue: found by the quarter only; grey: by the quarter and at least one month; light blue: "
                          "by one or more months only.",
                   sample=f"every chip of {SUB_TEXT}."),
        kit.figure(_c5, 5, "How many of its months caught each quarter detection.",
                   "Of the chips the quarter's picture detected, the share also detected in 0, 1, 2 or all 3 of its months. "
                   "Orange (0 of 3) are chips only the quarter found.",
                   mark="one quarter; segments add up to 100 %.", x="share of the quarter's detected chips.",
                   colour="number of the quarter's months that also detected the chip.",
                   sample=f"chips of {SUB_TEXT} detected in the quarter (rule as chosen above)."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A year from all four sources

    How detections pile up over one year in each source, and how much the sources agree on
    which chips are mines. AMW's yearly picture is one value for the whole year; the others
    accumulate period by period.
    """)
    return


@app.cell(hide_code=True)
def _(DET, mo):
    _years = sorted(int(y) for y in DET.loc[DET["year"] >= 2024, "year"].unique())
    year = mo.ui.dropdown(options=[str(y) for y in _years], value=str(_years[-1]) if _years else None, label="Year")
    year
    return (year,)


@app.cell(hide_code=True)
def _(DET, PERIODS_OF, SOURCES, SOURCE_COLOURS, SUB_TEXT, alt, at, kit, mo, pd, year):
    mo.stop(year.value is None, mo.md("No detections from 2024 on."))
    _y = int(year.value)
    _rows = []
    for _src in SOURCES:
        _tags = [t for t in PERIODS_OF[_src] if at.period_span(t)[0].year == _y]
        _acc = set()
        for _t in sorted(_tags, key=lambda t: at.period_span(t)[1]):
            _acc |= set(DET.loc[(DET["source"] == _src) & (DET["period"] == _t), "chip_id"])
            _rows.append({"source": _src, "up to": _t, "end": at.period_span(_t)[1] - pd.Timedelta(days=1), "chips": len(_acc)})
    _d = pd.DataFrame(_rows)
    mo.stop(_d.empty, mo.md(f"No source has periods in {_y}."))
    _scale = alt.Scale(domain=SOURCES, range=[SOURCE_COLOURS[s] for s in SOURCES])
    _base = alt.Chart(_d).encode(
        x=alt.X("end:T", title=f"end of the period ({_y})", scale=alt.Scale(domain=[f"{_y}-01-01", f"{_y}-12-31"])),
        y=alt.Y("chips:Q", title="cumulative unique detected chips"),
        color=alt.Color("source:N", sort=SOURCES, scale=_scale, title=None),
        tooltip=["source", "up to", alt.Tooltip("chips:Q", format=",")])
    _chart = alt.layer(_base.mark_line(interpolate="step-after"), _base.mark_point(filled=True, size=60)
                       ).properties(width="container", height=280)
    kit.figure(_chart, 6, f"Detections piling up over {_y}.",
               "For each source, the number of different chips detected from the start of the year up to the end of "
               "each period. A single dot is a source with one picture in the year (AMW yearly) or only one period "
               "assembled so far.",
               mark="one period of one source, placed at its last day.", x="last day of the period.",
               y="unique chips detected by that source from 1 January up to that period.", colour="source.",
               sample=f"every chip of {SUB_TEXT}.")
    return


@app.cell(hide_code=True)
def _(DET, SOURCES, SOURCE_COLOURS, SUB_TEXT, alt, kit, mo, pd, year):
    mo.stop(year.value is None)
    _y = int(year.value)
    _d = DET[DET["year"] == _y]
    _have = [s for s in SOURCES if (_d["source"] == s).any()]
    mo.stop(len(_have) < 2, mo.md(f"Fewer than two sources have detections in {_y}, so there is nothing to compare."))
    _m = pd.crosstab(_d["chip_id"], _d["source"]).reindex(columns=_have, fill_value=0) > 0
    _key = _m.apply(lambda r: "|".join(s for s in _have if r[s]), axis=1)
    _cnt = _key.value_counts().head(15)
    _order = list(_cnt.index)
    _bars = pd.DataFrame({"combination": _order, "chips": _cnt.to_numpy()})
    _dots = pd.DataFrame([{"combination": k, "source": s, "member": s in k.split("|")} for k in _order for s in _have])
    _x = alt.X("combination:N", sort=_order, title=None, axis=None)
    _top = alt.Chart(_bars).mark_bar(color="#555555").encode(
        x=_x, y=alt.Y("chips:Q", title="chips"), tooltip=["combination", alt.Tooltip("chips:Q", format=",")]
    ).properties(width=max(300, 60 * len(_order)), height=200)
    _labels = alt.Chart(_bars).mark_text(dy=-6, fontSize=11).encode(x=_x, y="chips:Q", text=alt.Text("chips:Q", format=","))
    _y_src = alt.Y("source:N", sort=_have, title=None)
    _w = max(300, 60 * len(_order))
    _matrix = alt.Chart(_dots).mark_circle(size=160, opacity=1).encode(
        x=_x, y=_y_src,
        color=alt.condition("datum.member", alt.Color("source:N", scale=alt.Scale(domain=_have, range=[SOURCE_COLOURS[s] for s in _have]),
                                                      legend=None), alt.value("#e5e5e5")),
        tooltip=["combination", "source", "member"],
    ).properties(width=_w, height=28 * len(_have))
    _chart = alt.vconcat(alt.layer(_top, _labels), _matrix, spacing=4).resolve_scale(x="shared")
    kit.figure(_chart, 7, f"Which sources agree on which chips, {_y}.",
               "Every chip detected at least once in the year by any source, grouped by the exact combination of sources "
               "that detected it. Each column is one combination: the filled dots below say which sources it contains, "
               "and the bar above how many chips have exactly that combination. The 15 most common combinations are shown.",
               mark="one combination of sources; bar height = number of chips with exactly that combination.",
               y="top: chips; bottom: the sources (filled = detected the chip at least once in the year).",
               sample=f"every chip of {SUB_TEXT}, detected in {_y} by at least one source.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Chip by chip: a quarter against one of its months

    Pick a quarter and one of its months. The scatter places every chip detected in either
    picture by its two scores. In the interactive notebook, drag a box over the scatter to list
    those chips in the table under it, with a link to the satellite view.
    """)
    return


@app.cell(hide_code=True)
def _(PERIODS, at, mo):
    _qs = [q for q in PERIODS["quarterly"] if any(m in PERIODS["monthly"] for m in at.months_of(q))]
    quarter = mo.ui.dropdown(options=_qs, value=_qs[0] if _qs else None, label="Quarter")
    return (quarter,)


@app.cell(hide_code=True)
def _(PERIODS, at, mo, quarter):
    _ms = [m for m in at.months_of(quarter.value) if m in PERIODS["monthly"]] if quarter.value else []
    month = mo.ui.dropdown(options=_ms, value=_ms[0] if _ms else None, label="Month")
    mo.stop(quarter.value is None, mo.md("No quarter has any of its months assembled."))
    mo.hstack([quarter, month], justify="start", gap=2)
    return (month,)


@app.cell(hide_code=True)
def _(CHIPS, MH, QH, at, month, mo, np, pd, quarter):
    mo.stop(month.value is None)
    _cols = ["chip_id", "confidence", "detected", "n_clear", "empty_share", "scored"]
    _q = QH.loc[QH["period"] == quarter.value, _cols].set_index("chip_id")
    _m = MH.loc[MH["period"] == month.value, _cols].set_index("chip_id")
    PAIR = _q.join(_m, lsuffix="_q", rsuffix="_m", how="inner")
    PAIR = PAIR[PAIR["scored_q"] & PAIR["scored_m"] & (PAIR["detected_q"] | PAIR["detected_m"])].reset_index()
    PAIR["group"] = np.where(PAIR["detected_q"] & PAIR["detected_m"], "both",
                             np.where(PAIR["detected_q"], "quarter only", "month only"))
    PAIR["month n_clear band"] = pd.cut(PAIR["n_clear_m"], at.NCLEAR_EDGES, right=False, labels=at.NCLEAR_LABELS).astype(str)
    PAIR = PAIR.merge(CHIPS, on="chip_id", how="left")
    PAIR["satellite view"] = [f"https://www.google.com/maps/@{la:.5f},{lo:.5f},16z/data=!3m1!1e3"
                              for la, lo in zip(PAIR["lat"], PAIR["lon"])]
    return (PAIR,)


@app.cell(hide_code=True)
def _(PAIR, RULE, alt, at, kit, mo, month, pd, quarter):
    GROUPS = ["both", "quarter only", "month only"]
    GROUP_COLOURS = ["#7f7f7f", kit.DETECTED, "#56B4E9"]
    _t = at.THRESHOLD[RULE]
    _brush = alt.selection_interval(encodings=["x", "y"])
    _shown, _thin_note = kit.thin(PAIR[["chip_id", "group", "confidence_q", "confidence_m", "n_clear_m", "empty_share_m"]]
                                  .astype({"chip_id": str}).round(3))
    _pts = alt.Chart(_shown).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("confidence_q:Q", title=f"confidence in the quarter ({quarter.value})", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("confidence_m:Q", title=f"confidence in the month ({month.value})", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", sort=GROUPS, scale=alt.Scale(domain=GROUPS, range=GROUP_COLOURS), title=None),
        tooltip=["chip_id", "group", alt.Tooltip("confidence_q:Q", format=".3f"), alt.Tooltip("confidence_m:Q", format=".3f"),
                 alt.Tooltip("n_clear_m:Q", format=".1f", title="n_clear (month)"),
                 alt.Tooltip("empty_share_m:Q", format=".2f", title="empty_share (month)")],
    ).add_params(_brush)
    _tt = pd.DataFrame({"t": [_t]})
    _rules = alt.layer(alt.Chart(_tt).mark_rule(strokeDash=[4, 4], color="#999").encode(x="t:Q"),
                       alt.Chart(_tt).mark_rule(strokeDash=[4, 4], color="#999").encode(y="t:Q"))
    scatter = mo.ui.altair_chart(alt.layer(_pts, _rules).properties(width=520, height=440))
    mo.vstack([scatter, kit.caption(8, f"Scores of {quarter.value} against {month.value}, chip by chip.",
               f"Every chip detected in the quarter, in the month, or in both, placed by its confidence in the quarter "
               f"(horizontal) and in the month (vertical). The dashed lines are the rule's threshold ({_t:g}); for the loose "
               f"and stringent rules an isolated chip also needs the higher score, so a few points beyond a line are not "
               f"detected. Quarter-only chips under the horizontal line are the month's misses: near the line they were close, "
               f"near 0 the month saw nothing.",
               mark="one chip.", x=f"confidence in {quarter.value}.", y=f"confidence in {month.value}.",
               colour="grey: detected in both; dark blue: quarter only; light blue: month only.",
               sample="the chosen subregions; chips scored in both pictures" + (f"; {_thin_note} (the table below and "
                      "Figures 9 and 10 use every chip)." if _thin_note else "."))])
    return GROUPS, GROUP_COLOURS, scatter


@app.cell(hide_code=True)
def _(PAIR, mo, scatter):
    _sel = scatter.apply_selection(PAIR)
    _boxed = len(_sel) < len(PAIR)
    _cols = ["chip_id", "group", "subregion", "country", "confidence_q", "confidence_m", "n_clear_q", "n_clear_m",
             "empty_share_q", "empty_share_m", "satellite view"]
    _t = _sel[_cols].sort_values(["group", "confidence_q"], ascending=[True, False]).reset_index(drop=True)
    _num = [c for c in _cols if c.startswith(("confidence", "n_clear", "empty_share"))]
    _t[_num] = _t[_num].astype(float).round(3)
    _t["chip_id"] = _t["chip_id"].astype(str)
    mo.vstack([mo.md(f"**{len(_sel):,} chips** {'in the box' if _boxed else '(drag a box on the scatter to narrow this down)'}. "
                     "Columns ending in _q are the quarter's values, _m the month's."),
               mo.ui.table(_t, page_size=10, selection=None)])
    return


@app.cell(hide_code=True)
def _(GROUPS, GROUP_COLOURS, PAIR, alt, at, kit, mo, month, np, pd, quarter):
    _bands = ["0", *at.NCLEAR_LABELS[1:]]
    _g = PAIR.groupby(["group", "month n_clear band"]).size().reset_index(name="chips")
    _g["share"] = _g["chips"] / _g.groupby("group")["chips"].transform("sum")
    _c1 = alt.Chart(_g).mark_bar().encode(
        y=alt.Y("group:N", sort=GROUPS, title=None),
        x=alt.X("share:Q", axis=alt.Axis(format="%"), title="share of the group's chips"),
        color=alt.Color("month n_clear band:N", sort=_bands, scale=alt.Scale(domain=_bands, range=["#E69F00", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]),
                        title=f"n_clear in {month.value}"),
        order=alt.Order("order:Q"),
        tooltip=["group", "month n_clear band", alt.Tooltip("chips:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
    ).transform_calculate(order=f"indexof({_bands}, datum['month n_clear band'])").properties(width="container", height=110)
    _counts, _edges = np.histogram(PAIR.loc[PAIR["group"] == "quarter only", "confidence_m"].dropna(), bins=40, range=(0, 1))
    _h = pd.DataFrame({"left": _edges[:-1], "right": _edges[1:], "chips": _counts})
    _c2 = alt.Chart(_h).mark_bar(color=kit.DETECTED).encode(
        x=alt.X("left:Q", title=f"confidence in {month.value}", scale=alt.Scale(domain=[0, 1])), x2="right:Q",
        y=alt.Y("chips:Q", title="chips"), y2=alt.datum(0), tooltip=["left", "right", alt.Tooltip("chips:Q", format=",")],
    ).properties(width="container", height=180)
    mo.vstack([
        kit.figure(_c1, 9, "How clear the month's view was, by group.",
                   f"For each group of Figure 8, the share of its chips in each band of n_clear (clear images per pixel) in "
                   f"{month.value}. If the month's misses come from clouds, the quarter-only row is dominated by the "
                   f"lowest bands while the 'both' row is not.",
                   mark="one group; segments add up to 100 %.", x="share of the group's chips.",
                   colour=f"n_clear band of the chip in {month.value}: orange = no clear image at all; blues, darker = more clear images.",
                   sample="the chips of Figure 8."),
        kit.figure(_c2, 10, "How far below the line the month's misses were.",
                   f"Confidence in {month.value} of the chips that only the quarter detected. Mass near 0 means the month "
                   f"gave no sign of mining; mass just under the threshold means the month nearly found them.",
                   mark="one bin of 0.025.", x=f"confidence in {month.value}.", y="number of quarter-only chips in the bin.",
                   sample=f"quarter-only chips of Figure 8 ({quarter.value} against {month.value})."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. When months miss, over all quarters

    Figures 8 to 10 look at one pair. These pool every quarter with assembled months: of the
    chips a quarter detected, how often each of its months also detected them, by how clear the
    month's view was.
    """)
    return


@app.cell(hide_code=True)
def _(COMPLETE_Q, MH, PERIODS, QH, SUB_TEXT, alt, at, kit, mo, pd):
    _rows = []
    for _q in PERIODS["quarterly"]:
        _qd = QH[(QH["period"] == _q) & QH["detected"]][["chip_id"]]
        for _pos, _m in enumerate(at.months_of(_q), start=1):
            if _m not in PERIODS["monthly"]:
                continue
            _mm = MH.loc[(MH["period"] == _m) & MH["scored"], ["chip_id", "detected", "n_clear"]]
            _rows.append(_qd.merge(_mm, on="chip_id").assign(quarter=_q, month=_m, position=f"month {_pos} of the quarter"))
    mo.stop(not _rows, mo.md("No quarter has any of its months assembled."))
    _d = pd.concat(_rows, ignore_index=True)
    _d["band"] = pd.cut(_d["n_clear"], at.NCLEAR_EDGES, right=False, labels=at.NCLEAR_LABELS).astype(str)
    _g = _d.groupby(["position", "band"]).agg(caught=("detected", "sum"), chips=("detected", "size")).reset_index()
    _all = _d.groupby("band").agg(caught=("detected", "sum"), chips=("detected", "size")).reset_index().assign(position="all months")
    _g = pd.concat([_g, _all], ignore_index=True)
    _g = _g[_g["chips"] >= 10]
    _lo, _hi = at.wilson(_g["caught"].to_numpy(), _g["chips"].to_numpy())
    _g["share"], _g["low"], _g["high"] = _g["caught"] / _g["chips"], _lo, _hi
    _pos = ["all months", "month 1 of the quarter", "month 2 of the quarter", "month 3 of the quarter"]
    _x = alt.X("band:N", sort=at.NCLEAR_LABELS, title="n_clear band in the month", axis=alt.Axis(labelAngle=0))
    _xo = alt.XOffset("position:N", sort=_pos)
    _col = alt.Color("position:N", sort=_pos, scale=alt.Scale(domain=_pos, range=["#222222", "#c6dbef", "#6baed6", "#08519c"]), title=None)
    _chart = alt.layer(
        alt.Chart(_g).mark_rule().encode(x=_x, xOffset=_xo, y=alt.Y("low:Q", title="share also detected in the month",
                                                                      axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                         y2="high:Q", color=_col),
        alt.Chart(_g).mark_point(filled=True, size=70).encode(
            x=_x, xOffset=_xo, y="share:Q", color=_col,
            tooltip=["position", "band", alt.Tooltip("caught:Q", format=","), alt.Tooltip("chips:Q", format=","),
                     alt.Tooltip("share:Q", format=".1%")]),
    ).properties(width="container", height=280)
    kit.figure(_chart, 11, "Share of quarter detections a month also finds, by the month's clear view.",
               "Of the chips detected in a quarter, the share also detected in one of its months, grouped by how many "
               "clear images per pixel (n_clear) that month had. A curve rising from left to right means months miss "
               "mainly when they are cloudy. Black pools all months; the blues split by the month's position in the quarter.",
               mark="one n_clear band (and month position) with at least 10 chip-months; the line is the 95 % Wilson interval.",
               x="n_clear band of the chip in the month.", y="share of quarter detections the month also detected.",
               colour="black: all months; light to dark blue: first, second, third month of the quarter.",
               sample=f"chips of {SUB_TEXT} detected in a quarter and scored in its month; every quarter with at least one "
                      f"month assembled.")
    return


@app.cell(hide_code=True)
def _(COMPLETE_Q, DET, alt, at, kit, mo, pd):
    mo.stop(not COMPLETE_Q, mo.md(""))
    _rows = []
    for (_sub, _src, _p), _g in DET[DET["source"].isin(["our months", "our quarters"])].groupby(["subregion", "source", "period"]):
        _rows.append({"subregion": _sub, "source": _src, "period": _p, "ids": set(_g["chip_id"])})
    _sets = {(r["subregion"], r["source"], r["period"]): r["ids"] for r in _rows}
    _out = []
    for _sub in sorted(DET["subregion"].unique()):
        for _q in COMPLETE_Q:
            _qs = _sets.get((_sub, "our quarters", _q), set())
            _ms = set().union(*[_sets.get((_sub, "our months", m), set()) for m in at.months_of(_q)])
            _n = len(_qs | _ms)
            if _n:
                _out.append({"subregion": str(_sub), "quarter": _q, "share": len(_qs - _ms) / _n,
                             "quarter only": len(_qs - _ms), "all": _n})
    _d = pd.DataFrame(_out)
    mo.stop(_d.empty, mo.md(""))
    _base = alt.Chart(_d).encode(x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
                                 y=alt.Y("subregion:N", title="subregion"))
    _chart = alt.layer(
        _base.mark_rect().encode(color=alt.Color("share:Q", scale=alt.Scale(scheme="oranges", domain=[0, max(0.01, _d["share"].max())]),
                                                 title="share found only by the quarter", legend=alt.Legend(format="%")),
                                 tooltip=["subregion", "quarter", alt.Tooltip("quarter only:Q", format=","),
                                          alt.Tooltip("all:Q", format=","), alt.Tooltip("share:Q", format=".1%")]),
        _base.mark_text(fontSize=12).encode(text=alt.Text("share:Q", format=".1%")),
    ).properties(width="container", height=40 * _d["subregion"].nunique())
    kit.figure(_chart, 12, "Where the quarter finds what its months miss.",
               "For each subregion and quarter, of all chips detected by the quarter or any of its months, the share that "
               "only the quarter's picture found. High values are where single months are not enough.",
               mark="one subregion and quarter; the number is the share.",
               colour="share of the chips found only by the quarter (darker = more).",
               sample="every chip of each subregion (the subregion filter above applies); quarters with all three months assembled.")
    return


if __name__ == "__main__":
    app.run()
