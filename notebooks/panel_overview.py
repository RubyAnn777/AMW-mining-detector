import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="The chip panel")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import geopandas as gpd
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    sys.path.insert(0, str(REPO / "panel_pipeline" / "checks"))
    import panel_views as pv
    from check_panel import check_panel
    from notebook_kit import MONTHLY, QUARTERLY, cached, can_write, find_runs, is_cached
    from periods import Period
    from variables import dictionary, expand

    _ = alt.data_transformers.disable_max_rows()
    FREQ_COLOURS = alt.Scale(domain=["monthly", "quarterly"], range=[MONTHLY, QUARTERLY])
    return (FREQ_COLOURS, Path, Period, REPO, alt, cached, can_write, check_panel, dictionary, expand,
            find_runs, gpd, is_cached, mo, np, os, pd, pq, pv)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The chip panel

    The data set the econometric work reads: one row per chip and period. A chip is AMW's
    48 × 48 pixel square (480 m on the ground), the unit the model scores; neighbouring chips
    overlap by half, 529 to a tile. There are two panels, monthly and quarterly, which share
    the chip IDs, so a month can be lined up with its quarter.

    The panel is balanced: every planned chip has a row in every period, scored or not. A
    chip whose tile could not be scored keeps its row with `scored = False` and empty values,
    so missing data never reads as "no mine".
    """)
    return


@app.cell(hide_code=True)
def _(find_runs, mo, os):
    _default = mo.cli_args().get("run") or os.environ.get("RUN") or ""
    _runs = find_runs(_default)
    _name = os.path.basename(str(_default).rstrip("/"))
    run_picker = (mo.ui.dropdown(options=_runs, value=_name if _name in _runs else None, label="Run", full_width=True)
                  if _runs else mo.ui.text(value=str(_default), label="Run folder", full_width=True))
    return (run_picker,)


@app.cell(hide_code=True)
def _(Path, mo, pv, run_picker):
    RUN = Path(run_picker.value or ".")
    _panels = pv.panels(RUN)
    panel_picker = mo.ui.dropdown(options=_panels, value="panel" if "panel" in _panels else (_panels[0] if _panels else None),
                                  label="Panel")
    mo.hstack([run_picker, panel_picker], widths=[3, 1])
    return RUN, panel_picker


@app.cell(hide_code=True)
def _(RUN, mo, panel_picker, pv):
    mo.stop(not panel_picker.value, mo.callout(mo.md(
        f"Nothing assembled in `{RUN}` yet (`cluster/assemble.sbatch` on Goethe)."), kind="warn"))
    PANEL = RUN / panel_picker.value
    periods = {f: pv.tags(PANEL, f) for f in pv.FREQUENCIES}
    RECORDS = pv.records(PANEL)
    mo.stop(not periods["quarterly"] and not periods["monthly"],
            mo.callout(mo.md(f"`{PANEL}` has no complete period yet."), kind="warn"))
    return PANEL, RECORDS, periods


@app.cell(hide_code=True)
def _(PANEL, RECORDS, RUN, can_write, is_cached, mo):
    _heavy = ["checks", "seen", "by_seen", "agree"]
    _missing = [n for n in _heavy if not is_cached(RUN, f"{n}_{PANEL.name}", RECORDS)]
    compute_now = mo.ui.run_button(label="Compute them now", kind="warn")
    (mo.callout(mo.vstack([
        mo.md(f"Some summaries of this panel have not been stored yet ({', '.join(_missing)}). Computing them "
              f"reads every period and takes a while on a full run; nothing is saved, since the run is read-only "
              f"for you. Ruby's `cluster/summaries.sbatch` stores them for everyone."),
        compute_now]), kind="neutral")
     if _missing and not can_write(RUN) else mo.md(""))
    return (compute_now,)


@app.cell(hide_code=True)
def _(RUN, can_write, compute_now):
    def may_compute():
        """Heavy summaries are computed by whoever can store them, or on request."""
        return can_write(RUN) or compute_now.value
    return (may_compute,)


@app.cell(hide_code=True)
def _(mo, periods):
    _q = periods["quarterly"]
    _arg = mo.cli_args().get("period")
    quarter = mo.ui.dropdown(options=_q, value=_arg if _arg in _q else (_q[-1] if _q else None), label="Quarter")
    rule = mo.ui.dropdown(options={"raw (confidence > 0.4)": "det_raw", "loose": "det_loose",
                                   "stringent": "det_stringent", "Andes": "det_andes"},
                          value="stringent", label="Detection rule")
    mo.vstack([mo.md("The quarter and rule below apply to the tables, the map and the distances further down."),
               mo.hstack([quarter, rule], justify="start", gap=2)])
    return quarter, rule


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What is in the panel

    Each period has two layers, two files with the same rows in the same order. `scores` holds
    what `detect.py` measured: the six model votes, their mean `confidence`, and how well the
    period was seen. `postprocess` holds AMW's detection flags and the distances behind them.
    What never changes about a chip (its square, its tile, whether it lies in the Andes) is
    stored once, in `plan/chips.parquet`.

    A tile on the border between two subregions is scored once by each. The panel keeps one
    row per chip with the higher of the two scores, as AMW does; "copies differ" counts how
    often the two disagreed.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, mo, pd, periods, pv):
    _rows = []
    for _f, _ts in periods.items():
        for _t in _ts:
            _s = pv.record(PANEL, _f, "scores", _t)
            _rows.append({"frequency": _f, "period": _t, "chips": _s["rows"], "scored": _s["scored"],
                          "subregions": ", ".join(map(str, _s["subregions"])),
                          "border chips": _s["border"]["chips_scored_twice"], "copies differ": _s["border"]["copies_differ"],
                          "MB": round(sum(pv.layer_path(PANEL, _f, l, _t).stat().st_size for l in pv.LAYERS) / 1e6)})
    mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The panel as a table

    The unit of observation is a chip in a period: `chip_id` is the cross-section, `period`
    the time dimension. A row is put together from three files that line up on `chip_id`:

    | part | file | holds |
    |:--|:--|:--|
    | the chip | `plan/chips.parquet` | location and fixed attributes |
    | `scores` | `<panel>/<frequency>/scores/period=<tag>/data.parquet` | the score and image quality |
    | `postprocess` | `<panel>/<frequency>/postprocess/period=<tag>/data.parquet` | AMW's flags and distances |

    Below are real rows of the chosen quarter: its strongest detections, a few random scored
    chips, and, where there are any, a chip with no clear image and a chip whose tile failed.
    Select rows to follow those chips through time in section 9. Every table can be
    downloaded as CSV from its bottom-right corner.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RUN, cached, np, pv, quarter):
    def _pick():
        _scored = pv.column(PANEL, "quarterly", "scores", quarter.value, "scored")
        _conf = pv.column(PANEL, "quarterly", "scores", quarter.value, "confidence")
        _strict = pv.column(PANEL, "quarterly", "postprocess", quarter.value, "det_stringent")
        _n_clear = pv.column(PANEL, "quarterly", "scores", quarter.value, "n_clear")
        _rng = np.random.default_rng(0)
        _top = [int(p) for p in np.argsort(np.where(_strict, _conf, -1))[::-1][:3] if _strict[p]]
        _picks = {p: "strongest stringent detection" for p in _top}
        _picks |= {int(p): "random scored chip" for p in _rng.choice(np.flatnonzero(_scored), min(3, int(_scored.sum())), replace=False)}
        for _p, _why in ((np.flatnonzero(_n_clear == 0)[:1], "no clear image (n_clear = 0)"),
                         (np.flatnonzero(~_scored)[:1], "tile failed (not scored)")):
            _picks |= {int(x): _why for x in _p}
        return _picks

    _rec = pv.layer_path(PANEL, "quarterly", "scores", quarter.value).parent / "_record.json"
    picks = cached(RUN, f"examples_{PANEL.name}_{quarter.value}", _pick, [_rec])
    return (picks,)


@app.cell(hide_code=True)
def _(PANEL, RUN, mo, np, picks, pv, quarter):
    _positions = np.array(list(picks))
    _rows = pv.period_rows(PANEL, "quarterly", quarter.value, _positions)
    _facts = pv.chip_facts(RUN, _rows["chip_id"])
    example = _rows.merge(_facts, on="chip_id", how="left")
    example.insert(0, "shown because", [picks[p] for p in _rows.index])
    example = example[["shown because", "chip_id", "period", "lon", "lat", "in_region", "in_andes",
                       *[c for c in _rows.columns if c not in ("chip_id", "period")],
                       "tile_key", "chip_index", "minx", "miny", "maxx", "maxy"]]
    example_table = mo.ui.table(example.astype({"chip_id": str} | {c: "float64" for c in example.select_dtypes("float32").columns}).round(4),
                                selection="multi", pagination=False,
                                label=f"{len(example)} rows of {quarter.value}, {example.shape[1] - 1} variables (scroll right)")
    example_table
    return (example_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What each variable means

    Written from the code that produces each variable (`plan.py`, `detect.py`,
    `assemble_period.py`); the source is `panel_pipeline/lib/variables.py`. "Missing when"
    matters for estimation: a missing value is never a zero. Use the search box to find a
    variable.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, dictionary, expand, mo, pv, quarter):
    _present = set(pv.PLAN_COLUMNS) | {"period", "lon", "lat", "amw_key_x", "amw_key_y"}
    for _layer in pv.LAYERS:
        _present |= set(pv.record(PANEL, "quarterly", _layer, quarter.value)["columns"])
    variables = dictionary()
    variables = variables[[any(n in _present for n in expand(v)) for v in variables["variable"]]]
    mo.ui.table(variables, selection=None, page_size=8, wrapped_columns=list(variables.columns[1:]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How each variable is distributed

    Over every chip of the chosen quarter and its first month. For a true/false variable the
    mean is the share that is true; "missing" is the share of chips without a value.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, Period, RUN, cached, mo, pd, periods, pv, quarter):
    _first = Period.parse(quarter.value).months()[0].tag
    _shown = [("quarterly", quarter.value)] + ([("monthly", _first)] if _first in periods["monthly"] else [])
    with mo.status.spinner(title="Reading every column of the period..."):
        _stats = pd.concat({_t: cached(RUN, f"describe_{PANEL.name}_{_t}", lambda f=_f, t=_t: pv.describe(PANEL, f, t),
                                       [pv.layer_path(PANEL, _f, "scores", _t).parent / "_record.json"])
                            for _f, _t in _shown}, axis=1)
    _stats.columns = [f"{t} {s}" for t, s in _stats.columns]
    mo.ui.table(_stats.round(3).reset_index(), selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Checks

    `check_panel.py` reads every period and tests what must always hold:

    | check | what must hold |
    |:--|:--|
    | rows | every planned chip once, sorted; both layers have the same rows |
    | scored | every chip has a score, or its tile is listed as failed |
    | votes | `confidence` is the mean of the six votes |
    | ranges | scores and shares between 0 and 1, `n_clear` ≤ `n_obs`, nothing infinite |
    | flags | `det_raw` is `confidence > 0.4`; stringent ⊂ loose ⊂ raw; flags empty exactly where there is no score |
    | distances | a distance exactly for the chips in each catalogue; the geodesic one within 10 % of AMW's |
    | border | the two scores of each border chip were identical |
    | months | a quarter's `n_obs` and `n_clear` are the sums of its three months' |
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RECORDS, RUN, cached, check_panel, is_cached, may_compute, mo, panel_picker):
    _name = f"checks_{PANEL.name}"
    mo.stop(not (is_cached(RUN, _name, RECORDS) or may_compute()), mo.md("*Not computed yet (see the note at the top).*"))
    _subregions = None if panel_picker.value == "panel" else [int(n) for n in panel_picker.value.split("_")[2:]]
    with mo.status.spinner(title="Checking every period..."):
        _report = cached(RUN, _name, lambda: check_panel(RUN, _subregions), RECORDS)
    _summary = _report.groupby("check", sort=False).agg(periods=("ok", "size"), passed=("ok", "sum"), example=("detail", "first"))
    _failed = _report[~_report["ok"]]
    mo.vstack([
        mo.callout(mo.md("All checks passed." if _failed.empty else f"**{len(_failed)} check(s) failed**, listed below."),
                   kind="success" if _failed.empty else "danger"),
        mo.ui.table(_summary.reset_index(), selection=None, pagination=False),
        mo.ui.table(_failed, selection=None) if len(_failed) else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. How many chips AMW's rules detect

    AMW's three rules, applied to every chip of every period:

    - **raw**: `confidence > 0.4`, the model's mean vote.
    - **loose** and **stringent**: a chip needs `confidence ≥ 0.43` (stringent: 0.55), or
      `≥ 0.75` (0.8) if it is isolated, meaning fewer than 5 other such chips within 3 km.
      Real mines come in clusters; a lone high score is more often a cloud edge or a river bank.

    A quarter's image is the median of three months of scenes, so it is usually clearer than
    any of its months, and a chip can be detected in the quarter but in none of its months.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, Period, mo, pd, periods, pv):
    _rows = []
    for _f, _ts in periods.items():
        for _t in _ts:
            _c = pv.record(PANEL, _f, "postprocess", _t)["counts"]
            _p = Period.parse(_t)
            _rows.append({"frequency": _f, "period": _t, "start": pd.Timestamp(_p.start),
                          "end": pd.Timestamp(_p.start) + pd.DateOffset(months=3 if _f == "quarterly" else 1),
                          "det_raw": _c["raw"], "det_loose": _c["loose"]["detected"],
                          "det_stringent": _c["stringent"]["detected"], "det_andes": _c["andes"],
                          "isolated (loose)": _c["loose"]["isolated"]})
    counts = pd.DataFrame(_rows)
    count_rule = mo.ui.radio(options={"raw": "det_raw", "loose": "det_loose", "stringent": "det_stringent", "Andes": "det_andes"},
                             value="stringent", inline=True, label="Rule")
    count_rule
    return count_rule, counts


@app.cell(hide_code=True)
def _(FREQ_COLOURS, alt, count_rule, counts, mo, pd):
    _r = count_rule.value
    _months = counts[counts["frequency"] == "monthly"].assign(mid=lambda d: d["start"] + pd.Timedelta(days=14))
    _quarters = counts[counts["frequency"] == "quarterly"]
    _tip = ["frequency", "period", alt.Tooltip(f"{_r}:Q", format=",", title="chips")]
    _m = alt.Chart(_months).mark_line(point=True).encode(
        x=alt.X("mid:T", title=None), y=alt.Y(f"{_r}:Q", title="chips detected", axis=alt.Axis(format=",d", tickMinStep=1), scale=alt.Scale(domainMin=0)),
        color=alt.Color("frequency:N", scale=FREQ_COLOURS, title=None), tooltip=_tip)
    _q = alt.Chart(_quarters).mark_rule(strokeWidth=3).encode(
        x="start:T", x2="end:T", y=f"{_r}:Q", color=alt.Color("frequency:N", scale=FREQ_COLOURS), tooltip=_tip)
    mo.vstack([
        (_m + _q).properties(width="container", height=280).interactive(bind_y=False),
        mo.md("<small>Each quarter is drawn as a line across its three months. Drag to pan, scroll to zoom in time.</small>"),
        mo.ui.table(counts.drop(columns=["start", "end"]), selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. How well each period was seen

    A chip's image is the median of the period's clear scenes (Cloud Score+ ≥ 0.6, AMW's cloud
    test). Where a pixel has no clear scene at all, the image is black there and the model sees
    nothing. `n_clear` is the average number of clear scenes per pixel of the chip.

    - *no clear image*: `n_clear = 0`, the whole chip was black to the model;
    - *under 1*: on average less than one clear scene per pixel;
    - *quality lost*: periods scored before 1 November 2026 lost `empty_share`, `ndvi_mean`,
      `bsi_mean`, `cirrus_mean` and `cs_clear_mean` for chips with some black pixels (a bug in
      `detect.py`, since fixed). The scores are not affected; those values are empty in the
      panel, never wrong.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RECORDS, RUN, cached, is_cached, may_compute, mo, np, pd, periods, pv):
    def _seen():
        _rows = []
        for _f, _ts in periods.items():
            for _t in _ts:
                _n_clear = pv.column(PANEL, _f, "scores", _t, "n_clear")
                _lost = pv.record(PANEL, _f, "scores", _t)["masked_pixels"]
                _rows.append({"frequency": _f, "period": _t, "median n_clear": float(np.nanmedian(_n_clear)),
                              "no clear image": float(np.mean(_n_clear == 0)), "under 1": float(np.mean(_n_clear < 1)),
                              "quality lost": _lost["chips_with_lost_quality_values"] / len(_n_clear)})
        return pd.DataFrame(_rows)

    _name = f"seen_{PANEL.name}"
    mo.stop(not (is_cached(RUN, _name, RECORDS) or may_compute()), mo.md("*Not computed yet (see the note at the top).*"))
    with mo.status.spinner(title="Reading n_clear of every period..."):
        seen = cached(RUN, _name, _seen, RECORDS)
    seen_measure = mo.ui.dropdown(options=["under 1", "no clear image", "quality lost", "median n_clear"],
                                  value="under 1", label="Show")
    seen_measure
    return seen, seen_measure


@app.cell(hide_code=True)
def _(FREQ_COLOURS, alt, mo, seen, seen_measure):
    _m = seen_measure.value
    _share = _m != "median n_clear"
    mo.vstack([
        alt.Chart(seen).mark_bar().encode(
            x=alt.X("period:N", sort=None, title=None),
            y=alt.Y(f"{_m}:Q", title=_m + (" (share of chips)" if _share else ""), axis=alt.Axis(format="%" if _share else "")),
            color=alt.Color("frequency:N", scale=FREQ_COLOURS, title=None),
            tooltip=["frequency", "period", alt.Tooltip(f"{_m}:Q", format=".1%" if _share else ".1f")],
        ).properties(width="container", height=260),
        mo.ui.table(seen.round(4), selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Do detections depend on how well a chip was seen?

    If the detector worked equally well under any sky, the share of chips detected would not
    change with `n_clear`. A chip with no clear image at all is a black square and should never
    be detected. Pooled over all periods of each frequency.

    If the share rises with `n_clear`, the detector finds fewer mines where it sees less, so a
    monthly count mixes mining with weather. That matters for any rule built on months
    (isolation, persistence), and it is why `n_clear` is in the panel as a control.
    """)
    return


@app.cell(hide_code=True)
def _(FREQ_COLOURS, PANEL, RECORDS, RUN, alt, cached, is_cached, may_compute, mo, np, pd, periods, pv):
    _edges = [0, 1e-9, 1, 3, 10, np.inf]
    _labels = ["none (0)", "under 1", "1 to 3", "3 to 10", "10 or more"]

    def _by_seen():
        _rows, _top = [], {}
        for _f, _ts in periods.items():
            _total, _hits, _black = np.zeros(5), np.zeros(5), 0.0
            for _t in _ts:
                _n_clear = pv.column(PANEL, _f, "scores", _t, "n_clear")
                _conf = pv.column(PANEL, _f, "scores", _t, "confidence")
                _raw = pv.column(PANEL, _f, "postprocess", _t, "det_raw")
                _k = np.digitize(np.nan_to_num(_n_clear, nan=-1), _edges) - 1      # -1: no score
                _ok = _k >= 0
                _total += np.bincount(_k[_ok], minlength=5)[:5]
                _hits += np.bincount(_k[_ok], weights=_raw[_ok], minlength=5)[:5]
                _black = max(_black, float(np.nanmax(_conf[_n_clear == 0], initial=0)))
            _rows += [{"frequency": _f, "n_clear": l, "chips": int(n), "detected (raw)": int(h),
                       "per 1,000 chips": 1000 * h / n if n else np.nan} for l, n, h in zip(_labels, _total, _hits)]
            _top[_f] = _black
        return pd.DataFrame(_rows), _top

    _name = f"by_seen_{PANEL.name}"
    mo.stop(not (is_cached(RUN, _name, RECORDS) or may_compute()), mo.md("*Not computed yet (see the note at the top).*"))
    with mo.status.spinner(title="Reading every period..."):
        _table, _top = cached(RUN, _name, _by_seen, RECORDS)
    mo.vstack([
        alt.Chart(_table).mark_bar().encode(
            x=alt.X("n_clear:N", sort=_labels, title="clear scenes per pixel (n_clear)"),
            xOffset="frequency:N", y=alt.Y("per 1,000 chips:Q", title="raw detections per 1,000 chips"),
            color=alt.Color("frequency:N", scale=FREQ_COLOURS, title=None),
            tooltip=["frequency", "n_clear", alt.Tooltip("chips:Q", format=","), alt.Tooltip("per 1,000 chips:Q", format=".2f")],
        ).properties(width="container", height=260),
        mo.md(" ".join(f"Highest score of a {f} chip with no clear image: {v:.3f}." for f, v in _top.items())),
        mo.ui.table(_table.round(3), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Do the months agree with their quarter?

    For each quarter whose three months are assembled: of the chips detected in the quarter,
    the share also detected in at least one of its months and in all three; and the other way
    round. A mine detected in its quarter but not in every month was often just not visible
    that month; section 9 shows examples.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, Period, RECORDS, RUN, cached, is_cached, may_compute, mo, np, pd, periods, pv):
    def _agree():
        _rows = []
        for _q in periods["quarterly"]:
            _months = [m.tag for m in Period.parse(_q).months()]
            if not all(m in periods["monthly"] for m in _months):
                continue
            for _rule in ("det_raw", "det_stringent"):
                _in_q = pv.column(PANEL, "quarterly", "postprocess", _q, _rule)
                _in_m = np.stack([pv.column(PANEL, "monthly", "postprocess", m, _rule) for m in _months])
                _any, _all = _in_m.any(axis=0), _in_m.all(axis=0)
                _rows.append({"quarter": _q, "rule": _rule.removeprefix("det_"), "in quarter": int(_in_q.sum()),
                              "also in a month": (_in_q & _any).sum() / max(_in_q.sum(), 1),
                              "in all 3 months": (_in_q & _all).sum() / max(_in_q.sum(), 1),
                              "in a month": int(_any.sum()),
                              "also in the quarter": (_any & _in_q).sum() / max(_any.sum(), 1)})
        return pd.DataFrame(_rows)

    _name = f"agree_{PANEL.name}"
    mo.stop(not (is_cached(RUN, _name, RECORDS) or may_compute()), mo.md("*Not computed yet (see the note at the top).*"))
    with mo.status.spinner(title="Comparing months with quarters..."):
        _agree_table = cached(RUN, _name, _agree, RECORDS)
    mo.ui.table(_agree_table.round(3), selection=None, pagination=False) if len(_agree_table) \
        else mo.md("No quarter has all three of its months assembled yet.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Where the detections are

    Every chip detected in the chosen quarter by the chosen rule, over the subregion outlines.
    Scroll to zoom, drag to pan, and click a chip (shift-click for several) to follow it in
    section 9.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, REPO, RUN, alt, gpd, mo, pd, pq, pv, quarter, rule):
    _path = pv.layer_path(PANEL, "quarterly", "postprocess", quarter.value)
    _flags = pq.read_table(_path, columns=["chip_id", rule.value])
    _ids = _flags.filter(_flags[rule.value].fill_null(False))["chip_id"].to_numpy()
    _scores = pv.rows_of_chips(PANEL, "quarterly", quarter.value, _ids, {"scores": ["confidence", "n_clear", "subregion"]})
    _where = pv.chip_facts(RUN, _ids)[["chip_id", "lon", "lat"]]
    detected = _scores.merge(_where, on="chip_id")

    _rings = []
    for _n in pv.record(PANEL, "quarterly", "scores", quarter.value)["subregions"]:
        _file = REPO / "data" / "boundaries" / "Amazon_ACA" / f"Amazon_ACA_{_n}.geojson"
        if not _file.is_file():
            continue
        for _poly in gpd.read_file(_file).explode(index_parts=False).simplify(0.01):
            _x, _y = _poly.exterior.xy
            _rings.append(pd.DataFrame({"lon": _x, "lat": _y, "ring": f"{_n}-{len(_rings)}", "order": range(len(_x))}))
    _scale = dict(scale=alt.Scale(zero=False))
    _pick = alt.selection_point(name="chips", fields=["chip_id"], on="click", toggle="event.shiftKey")
    _points = alt.Chart(detected).mark_circle(size=14, color="#eb6834").add_params(_pick).encode(
        x=alt.X("lon:Q", title="longitude", **_scale), y=alt.Y("lat:Q", title="latitude", **_scale),
        tooltip=["chip_id", "subregion", alt.Tooltip("confidence:Q", format=".3f"), alt.Tooltip("n_clear:Q", format=".1f")])
    _layers = [_points]
    if _rings:
        _layers.insert(0, alt.Chart(pd.concat(_rings)).mark_line(color="#8a8984", strokeWidth=0.8).encode(
            x=alt.X("lon:Q", **_scale), y=alt.Y("lat:Q", **_scale), detail="ring:N", order="order:Q"))
    _rule_name = [k for k, v in rule.options.items() if v == rule.value][0]
    map_chart = mo.ui.altair_chart(
        alt.layer(*_layers).properties(width="container", height=520,
                                       title=f"{len(detected):,} chips detected ({_rule_name}), {quarter.value}").interactive(),
        chart_selection=False, legend_selection=False)
    map_chart
    return (map_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Following chips through time

    Every period holds the same chips, so a chip can be followed from month to month and
    quarter to quarter. The chips shown are the ones selected in the table of section 2 or on
    the map, plus any IDs typed here. With nothing selected, the five strongest detections of
    the chosen quarter are shown.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    chip_input = mo.ui.text(placeholder="e.g. 205105103000, 205105103001", label="Chip IDs", full_width=True)
    chip_input
    return (chip_input,)


@app.cell(hide_code=True)
def _(PANEL, chip_input, detected, example_table, map_chart, mo, np, pv, quarter):
    _ids = []
    _ids += [int(x) for x in getattr(example_table.value, "chip_id", [])]   # text in the table
    _sel = map_chart.apply_selection(detected)
    if len(_sel) < len(detected):                    # nothing clicked = everything "selected"
        _ids += [int(x) for x in _sel["chip_id"]]
    _ids += [int(x) for x in chip_input.value.replace(",", " ").split() if x.isdigit()]
    if not _ids:
        _conf = pv.column(PANEL, "quarterly", "scores", quarter.value, "confidence")
        _det = pv.column(PANEL, "quarterly", "postprocess", quarter.value, "det_stringent")
        _chip = pv.column(PANEL, "quarterly", "scores", quarter.value, "chip_id")
        _ids = [int(_chip[p]) for p in np.argsort(np.where(_det, _conf, -1))[::-1][:5]]
    follow = list(dict.fromkeys(_ids))[:20]
    mo.md(f"Following {len(follow)} chip(s)" + (" (the first 20 of the selection)." if len(_ids) > 20 else "."))
    return (follow,)


@app.cell(hide_code=True)
def _(FREQ_COLOURS, PANEL, Period, alt, follow, mo, pd, periods, pv):
    _cols = {"scores": ["scored", "confidence", "n_obs", "n_clear", "empty_share", "cs_mean", "ndvi_mean", "bsi_mean"],
             "postprocess": ["det_raw", "det_loose", "det_stringent"]}
    with mo.status.spinner(title="Reading the chips in every period..."):
        _parts = [pv.rows_of_chips(PANEL, _f, _t, follow, _cols).assign(frequency=_f) for _f, _ts in periods.items() for _t in _ts]
    history = pd.concat(_parts, ignore_index=True)
    history["start"] = [pd.Timestamp(Period.parse(t).start) for t in history["period"]]
    history["middle"] = history["start"] + pd.to_timedelta(history["frequency"].map({"monthly": 14, "quarterly": 45}), unit="D")
    history["chip"] = history["chip_id"].astype(str)
    history = history.sort_values(["chip_id", "frequency", "start"]).reset_index(drop=True)
    _line = alt.Chart(history).mark_line(point=True).encode(
        x=alt.X("middle:T", title=None), y=alt.Y("confidence:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("chip:N", title="chip"), strokeDash=alt.StrokeDash("frequency:N", title=None),
        detail="frequency:N",
        tooltip=["chip", "frequency", "period", alt.Tooltip("confidence:Q", format=".3f"),
                 alt.Tooltip("n_clear:Q", format=".1f"), "det_stringent"])
    _cut = alt.Chart().mark_rule(color="grey", strokeDash=[2, 2]).encode(y=alt.datum(0.4))
    mo.vstack([
        (_line + _cut).properties(width="container", height=300, title="Confidence through time (dashed line: 0.4)").interactive(bind_y=False),
        mo.ui.table(history.drop(columns=["start", "middle", "chip_id"]).rename(columns={"chip": "chip_id"})
                    .pipe(lambda d: d.astype({c: "float64" for c in d.select_dtypes("float32").columns})).round(3),
                    selection=None, page_size=15),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. AMW's distances against distances on the Earth

    To decide whether a chip is isolated, AMW projects all chips of the period onto one flat
    map (the UTM zone in the middle of them) and measures distances there. Far from that zone
    the map stretches distances. The panel stores both AMW's distance (`kth_km_*`, used by the
    flags) and the distance on the Earth's real shape (`kth_km_geo_*`), so the isolation rule
    can be revisited. The chart shows the ratio for the loose catalogue of the chosen quarter.
    """)
    return


@app.cell(hide_code=True)
def _(PANEL, RUN, alt, mo, pq, pv, quarter):
    _pp = pq.read_table(pv.layer_path(PANEL, "quarterly", "postprocess", quarter.value),
                        columns=["chip_id", "kth_km_loose", "kth_km_geo_loose"]).to_pandas()
    _cat = _pp[_pp["kth_km_loose"].notna()]
    mo.stop(_cat.empty, mo.md(f"No chip of {quarter.value} is in the loose catalogue."))
    _cat = _cat.merge(pv.chip_facts(RUN, _cat["chip_id"])[["chip_id", "lon"]], on="chip_id")
    _cat["ratio"] = _cat["kth_km_loose"] / _cat["kth_km_geo_loose"]
    _flip = int(((_cat["kth_km_loose"] > 3.0) != (_cat["kth_km_geo_loose"] > 3.0)).sum())
    mo.vstack([
        (alt.Chart(_cat).mark_circle(size=12, opacity=0.5, color="#2a78d6").encode(
            x=alt.X("lon:Q", title="longitude", scale=alt.Scale(zero=False)),
            y=alt.Y("ratio:Q", title="AMW's distance / distance on the Earth", scale=alt.Scale(zero=False)),
            tooltip=["chip_id", alt.Tooltip("kth_km_loose:Q", format=".2f"), alt.Tooltip("kth_km_geo_loose:Q", format=".2f")])
         + alt.Chart().mark_rule(color="#8a8984").encode(y=alt.datum(1))).properties(width="container", height=260).interactive(),
        mo.md(f"{len(_cat):,} chips in the catalogue; the ratio runs from {_cat['ratio'].min():.3f} to {_cat['ratio'].max():.3f}. "
              f"{_flip} would switch between isolated and not if the rule used the distance on the Earth."),
    ])
    return


if __name__ == "__main__":
    app.run()
