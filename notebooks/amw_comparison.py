import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Our run against AMW")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "amw_comparison"))
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    import compare
    from notebook_kit import find_runs

    _ = alt.data_transformers.disable_max_rows()
    return Path, alt, compare, find_runs, json, mo, np, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Our run against AMW, chip by chip

    Does our `detect.py` reproduce the quarterly detections AMW publishes? Both sides score
    the same 48 px chips, so every chip AMW lists can be looked up in our run. AMW publishes
    only chips with a confidence above 0.4 and only quarters, so the comparison is about
    detections, not areas, and our months are compared with AMW's quarter.

    The join itself runs on the cluster (`cluster/amw_compare.sbatch`) and writes a small
    results folder, `<run>/amw_comparison/`. This notebook only reads that folder.
    """)
    return


@app.cell(hide_code=True)
def _(find_runs, mo, os):
    _default = mo.cli_args().get("run") or os.environ.get("RUN") or ""
    _runs = find_runs(_default)
    _name = os.path.basename(str(_default).rstrip("/"))
    run_picker = (mo.ui.dropdown(options=_runs, value=_name if _name in _runs else None, label="Run", full_width=True)
                  if _runs else mo.ui.text(value=str(_default), label="Run folder", full_width=True))
    run_picker
    return (run_picker,)


@app.cell(hide_code=True)
def _(Path, compare, mo, pd, run_picker):
    RUN = Path(run_picker.value or ".")
    RESULTS = RUN / "amw_comparison"
    _available = sorted(p.stem.removeprefix("chips_") for p in RESULTS.glob("chips_*.parquet"))
    mo.stop(not _available, mo.callout(mo.md(
        f"No comparison results in `{RESULTS}`. They are made on Goethe with "
        f"`sbatch cluster/amw_compare.sbatch $RUN <year>`."), kind="warn"))
    chips = {q: pd.read_parquet(RESULTS / f"chips_{q}.parquet") for q in _available}
    QUARTERS = list(chips)
    MONTH_COLUMNS = {q: [c for c in chips[q] if c.startswith("conf_")] for q in QUARTERS}
    RAW = compare.RAW_THRESHOLD
    return MONTH_COLUMNS, QUARTERS, RAW, RESULTS, RUN, chips


@app.cell(hide_code=True)
def _(QUARTERS, RAW, chips, mo):
    _subregions = sorted({int(s) for d in chips.values() for s in d["subregion"].unique()})
    quarter = mo.ui.dropdown(options=QUARTERS, value=QUARTERS[-1], label="Quarter")
    threshold = mo.ui.slider(RAW, 0.95, step=0.01, value=RAW, label="Detection threshold", show_value=True)
    subregion_filter = mo.ui.multiselect(options=[str(s) for s in _subregions], label="Subregions (empty = all)")
    mo.vstack([
        mo.hstack([quarter, threshold, subregion_filter], justify="start", gap=2),
        mo.md(f"<small>AMW's threshold is {RAW}: a chip counts as detected above it. AMW's files "
              f"list nothing below {RAW}, so the slider starts there. Raising it shows whether the "
              f"disagreements are borderline scores or confident ones.</small>"),
    ])
    return quarter, subregion_filter, threshold


@app.cell(hide_code=True)
def _(MONTH_COLUMNS, chips, compare, subregion_filter, threshold):
    def selected(q):
        """The chips of quarter q in the chosen subregions, with outcomes at the chosen threshold."""
        d = chips[q]
        if subregion_filter.value:
            d = d[d["subregion"].astype(str).isin(subregion_filter.value)]
        d = d.assign(outcome=compare.outcome(d, threshold.value))
        if MONTH_COLUMNS[q]:
            d["months_detected"] = (d[MONTH_COLUMNS[q]] > threshold.value).sum(axis=1)
        return d

    view = {q: selected(q) for q in chips}
    return (view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agreement

    For each quarter, how many chips both sides detect, how many only one side does, and the
    overlap: chips both detected divided by chips either detected, so 1 means the same set.
    """)
    return


@app.cell(hide_code=True)
def _(compare, mo, pd, threshold, view):
    _whole = ["both", "only ours", "only AMW", "ours total", "AMW total"]
    summary = pd.DataFrame({q: compare.agreement(d, threshold.value) for q, d in view.items()}).T.drop(columns="neither")
    summary[_whole] = summary[_whole].astype(int)
    _per_sub = pd.concat({q: d.groupby("subregion").apply(compare.agreement, threshold.value, include_groups=False)
                          for q, d in view.items()}, names=["quarter"]).drop(columns="neither")
    _per_sub[_whole] = _per_sub[_whole].astype(int)
    mo.ui.tabs({
        "per quarter": mo.ui.table(summary.round(4).rename_axis("quarter").reset_index(), selection=None, pagination=False),
        "per subregion": mo.ui.table(_per_sub.round(4).reset_index(), selection=None),
    })
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scores

    Where both sides detect a chip, how far apart are the two scores? A gap of about 1e-7 is
    rounding and means the same images went in. Points off the diagonal are chips where the
    image itself differed.
    """)
    return


@app.cell(hide_code=True)
def _(alt, compare, mo, pd, quarter, threshold, view):
    _gaps = pd.DataFrame({q: compare.score_gap(d) for q, d in view.items()}).T.rename_axis("quarter").reset_index()
    _d = view[quarter.value]
    _sample = _d[_d["amw_confidence"].notna()]
    _scatter = (alt.Chart(_sample).mark_circle(size=22, opacity=0.6)
                .encode(x=alt.X("amw_confidence:Q", title="AMW confidence", scale=alt.Scale(domain=[0.35, 1])),
                        y=alt.Y("confidence:Q", title="our confidence", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("outcome:N", scale=alt.Scale(domain=["both", "only ours", "only AMW"],
                                                                      range=["#8a8984", "#2a78d6", "#d64a2a"])),
                        tooltip=["chip_id", "subregion", alt.Tooltip("confidence:Q", format=".4f"),
                                 alt.Tooltip("amw_confidence:Q", format=".4f")]))
    _diagonal = alt.Chart(pd.DataFrame({"x": [0, 1]})).mark_line(color="black", strokeDash=[4, 4], strokeWidth=0.7).encode(x="x", y="x")
    _cut = (alt.Chart().mark_rule(color="grey", strokeWidth=0.6).encode(y=alt.datum(threshold.value))
            + alt.Chart().mark_rule(color="grey", strokeWidth=0.6).encode(x=alt.datum(threshold.value)))
    _dis = _d[_d["outcome"].isin(["only ours", "only AMW"])]
    _near = int(((_dis["confidence"] - threshold.value).abs() <= 0.05).sum())
    mo.vstack([
        mo.ui.table(_gaps.round(6), selection=None, pagination=False),
        (_scatter + _diagonal + _cut).properties(width=420, height=380, title=quarter.value).interactive(),
        mo.md(f"{quarter.value}: {len(_dis):,} chips where only one side detects; for {_near:,} of them our "
              f"score is within 0.05 of the threshold."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image quality and disagreement

    Median quality per outcome. A quarter's image is a median of the scenes in it, so a chip
    with few clear looks (`n_clear`) depends on one or two scenes, and any difference in which
    scenes went in moves its score more.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd, view):
    _cols = [c for c in ["confidence", "amw_confidence", "n_obs", "n_clear"] if all(c in d for d in view.values())]
    _t = pd.concat({q: d.groupby("outcome")[_cols].median() for q, d in view.items()}, names=["quarter"])
    mo.ui.table(_t.round(3).reset_index(), selection=None, pagination=False) if len(_cols) > 2 \
        else mo.md("This run's comparison files have no quality columns.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the scores differ: the last day of the quarter

    AMW's quarters stop one day early. Earth Engine treats the end date as excluded, and AMW
    passes the quarter's last day as the end, so 31 Mar, 30 Jun, 30 Sep and 31 Dec never enter
    AMW's images. Our quarters end on the first of the next month, so every day is in. The
    dates come from AMW's file name and our task's `_status.json`.

    On Pará, 2026Q1, this was settled: AMW's own code with AMW's end date reproduces AMW's
    published file (27 of 27 chips, identical scores), and with our end date it reproduces our
    scores (56 of 56). Our code is AMW's code; only the end date differs. We keep the last day,
    so our quarters are whole calendar quarters like our months.
    """)
    return


@app.cell(hide_code=True)
def _(QUARTERS, RESULTS, RUN, json, mo, pd):
    _files = pd.read_csv(RESULTS / "amw_files.csv").set_index("quarter")
    _rows = {}
    for _q in QUARTERS:
        _name = _files.loc[_q, "file"] if _q in _files.index else ""
        _task = next(RUN.glob(f"detect/quarterly/period={_q}/subregion=*/_status.json"), None)
        _ours = json.loads(_task.read_text()) if _task else {}
        _rows[_q] = {"AMW file": _name, "AMW rows": _files.loc[_q, "rows"] if _q in _files.index else None,
                     "fingerprint": _files.loc[_q, "sha256"] if _q in _files.index else None,
                     "our start": _ours.get("start_date"), "our end (excluded)": _ours.get("end_date_excluded")}
    _checks = {_q: RESULTS / "last_day_check" / _q / "summary.txt" for _q in QUARTERS}
    _found = {q: p.read_text() for q, p in _checks.items() if p.is_file()}
    mo.vstack([
        mo.ui.table(pd.DataFrame(_rows).T.rename_axis("quarter").reset_index(), selection=None, pagination=False),
        mo.md("**The last-day test, per quarter** (from `cluster/last_day_check.sbatch`):") if _found else
        mo.md("The last-day test has not been run for these quarters (`sbatch cluster/last_day_check.sbatch $RUN <year>` on Goethe)."),
        mo.accordion({q: mo.plain_text(t) for q, t in _found.items()}) if _found else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where they differ

    Every chip someone detected in the chosen quarter. Grey: both detect it with the same score.
    Coloured: both detect it, with different scores. Blue and red: only we, or only AMW, detect
    it. Scroll to zoom, drag to pan, hover for the chip.
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, np, quarter, view):
    _d = view[quarter.value]
    mo.stop(not {"amw_key_x", "amw_key_y"} <= set(_d.columns),
            mo.md("This run's comparison files have no chip locations (made before `amw_key_x/y` were kept)."))
    _d = _d[_d["outcome"] != "neither"].assign(gap=(_d["confidence"] - _d["amw_confidence"]).abs())
    _d["kind"] = np.select([(_d["outcome"] == "both") & (_d["gap"] <= 1e-6), _d["outcome"] == "both"],
                           ["same score", "different score"], _d["outcome"])
    _counts = _d["kind"].value_counts()
    _d["legend"] = _d["kind"].map(lambda k: f"{k} ({_counts[k]:,})")
    _order = [f"{k} ({_counts[k]:,})" for k in ["same score", "different score", "only ours", "only AMW"] if k in _counts]
    mo.ui.altair_chart(
        alt.Chart(_d).mark_circle(size=14)
        .encode(x=alt.X("amw_key_x:Q", title="longitude", scale=alt.Scale(zero=False)),
                y=alt.Y("amw_key_y:Q", title="latitude", scale=alt.Scale(zero=False)),
                color=alt.Color("legend:N", title=None, sort=_order,
                                scale=alt.Scale(domain=_order, range=["#a3a29c", "#e2a400", "#2a78d6", "#d64a2a"][:len(_order)])),
                tooltip=["chip_id", "subregion", "kind", alt.Tooltip("confidence:Q", format=".4f"),
                         alt.Tooltip("amw_confidence:Q", format=".4f"), alt.Tooltip("gap:Q", format=".2e")])
        .properties(width="container", height=480).interactive(),
        chart_selection=False, legend_selection=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Border chips

    A tile on the line between two subregions is in both subregions' lists and is scored twice,
    by us and by AMW. If both of our copies have the same score, they saw the same images, and
    a gap to AMW at the border has the same cause as anywhere else.
    """)
    return


@app.cell(hide_code=True)
def _(compare, mo, np, pd, threshold, view):
    _rows = {}
    for _q, _d in view.items():
        if "border_copies" not in _d:
            continue
        _border = _d["border_copies"] > 1
        for _name, _part in (("border", _d[_border]), ("elsewhere", _d[~_border])):
            if _part.empty:
                continue
            _a, _g = compare.agreement(_part, threshold.value), compare.score_gap(_part)
            _rows[(_q, _name)] = {"chips": len(_part), "both": int(_a["both"]), "only ours": int(_a["only ours"]),
                                  "only AMW": int(_a["only AMW"]), "overlap": _a["overlap (Jaccard)"],
                                  "same score as AMW": _g["identical (gap <= 1e-6)"],
                                  "our two copies identical": (_part["border_spread"] <= 1e-6).mean() if _name == "border" else np.nan,
                                  "largest gap between our copies": _part["border_spread"].max() if _name == "border" else np.nan}
    mo.ui.table(pd.DataFrame(_rows).T.round(4).rename_axis(["quarter", "where"]).reset_index(), selection=None, pagination=False) \
        if _rows else mo.md("No chip was scored twice: border chips need both neighbouring subregions in the run.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Our months against AMW's quarter

    A mine visible all quarter should be detected in most of its months; clouds make single
    months miss. "Any month" is the union of the three.
    """)
    return


@app.cell(hide_code=True)
def _(MONTH_COLUMNS, mo, pd, threshold, view):
    _rows, _by_months = {}, {}
    for _q, _d in view.items():
        if not MONTH_COLUMNS[_q]:
            continue
        _amw, _ours, _any = _d["amw_confidence"] > threshold.value, _d["confidence"] > threshold.value, _d["months_detected"] > 0
        _rows[_q] = {"months scored": len(MONTH_COLUMNS[_q]), "AMW": int(_amw.sum()), "any month": int(_any.sum()),
                     "AMW and a month": int((_amw & _any).sum()), "AMW, no month": int((_amw & ~_any).sum()),
                     "a month, not AMW": int((_any & ~_amw).sum()),
                     "overlap AMW / any month": (_amw & _any).sum() / max((_amw | _any).sum(), 1),
                     "overlap our quarter / any month": (_ours & _any).sum() / max((_ours | _any).sum(), 1)}
        _by_months[_q] = _d.loc[_amw, "months_detected"].value_counts().reindex(range(4), fill_value=0)
    mo.vstack([
        mo.ui.table(pd.DataFrame(_rows).T.round(3).rename_axis("quarter").reset_index(), selection=None, pagination=False),
        mo.md("AMW's detections by how many of our months also detect them:"),
        mo.ui.table(pd.DataFrame(_by_months).rename_axis("months detected").reset_index(), selection=None, pagination=False),
    ]) if _rows else mo.md("No months were scored for these quarters.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs

    Every task the join read has to be complete with no failed tiles, because a missing tile
    looks exactly like "AMW detected it, we didn't". "Should have matched" counts AMW chips
    that fall between our chips; it must be 0, or the two grids differ. "Chips sharing a key"
    counts the same ground in two tiles of one subregion (where UTM zones meet); as AMW does,
    the higher score is kept.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, pd):
    _inputs = pd.read_csv(RESULTS / "inputs.csv")
    _scored = _inputs[_inputs["state"] != "missing"]
    _problems = _scored[(_scored["state"] != "complete") | (_scored["failed"] > 0)]
    mo.vstack([
        mo.md(f"{len(_scored)} tasks read, {len(_problems)} of them incomplete or with failed tiles."),
        mo.ui.table(_problems, selection=None) if len(_problems) else mo.md(""),
        mo.ui.table(pd.read_csv(RESULTS / "joins.csv"), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, quarter, summary, threshold):
    _c = summary.loc[quarter.value]
    mo.callout(mo.md(
        f"**{quarter.value}** at {threshold.value:.2f}: we detect {_c['ours total']:,} chips, AMW {_c['AMW total']:,}. "
        f"Both detect {_c['both']:,}, only we {_c['only ours']:,}, only AMW {_c['only AMW']:,}; "
        f"overlap {_c['overlap (Jaccard)']:.3f}."), kind="info")
    return


if __name__ == "__main__":
    app.run()
