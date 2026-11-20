import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Only AMW's mining tiles?")


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
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    from notebook_kit import thin

    _ = alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, np, os, pd, thin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Could we score only the tiles where AMW has found mining?

    The idea: AMW publishes a cumulative record of every place it has found mining since 2018.
    If new mines almost always appear in or next to tiles that are already in that
    record, we could score those tiles first (priority 1: AMW's tiles plus a ring of
    neighbours) and the rest of the basin later (priority 2), and save most of the Earth
    Engine budget on the first pass.

    **The test.** For 2025 and 2026 we already scored the whole basin. So we can ask: of the
    chips we detected in a quarter, how many lie in a tile where AMW had found mining
    *before* that year, or within one, two or three tiles of one? Whatever lies further out
    is what priority 1 alone would have missed.

    A tile is 5.52 km a side (552 pixels of 10 m) and holds 529 chips. "Ring 1" is the eight
    tiles around an AMW tile, "ring 2" the next ring out, and so on.
    """)
    return


@app.cell(hide_code=True)
def _(Path, json, mo, os, pd):
    RUN = Path(mo.cli_args().get("run") or os.environ.get("RUN") or ".").expanduser()   # view.sbatch passes --run
    RESULTS = RUN / "amw_comparison"
    mo.stop(not (RESULTS / "history_tiles.parquet").is_file(), mo.callout(mo.md(
        f"No AMW history tiles in `{RESULTS}` yet. They are made on Goethe with "
        f"`sbatch cluster/amw_history.sbatch $RUN` (a few minutes)."), kind="warn"))
    tiles = pd.read_parquet(RESULTS / "history_tiles.parquet")
    SUMMARY = json.loads((RESULTS / "history_summary.json").read_text())
    _available = sorted(p.stem.removeprefix("chips_") for p in RESULTS.glob("chips_*.parquet"))
    QUARTERS = [q for q in _available if q[:4] >= "2025"]
    mo.stop(not QUARTERS, mo.callout(mo.md(
        f"No compared quarters from 2025 on in `{RESULTS}`. They come from "
        f"`sbatch cluster/amw_compare.sbatch $RUN 2025` (and 2026)."), kind="warn"))
    chips = {q: pd.read_parquet(RESULTS / f"chips_{q}.parquet",
                                columns=["chip_id", "subregion", "amw_key_x", "amw_key_y", "confidence", "amw_confidence"])
             for q in QUARTERS}
    CUTOFFS = SUMMARY["cutoffs"]
    MAX_RING = SUMMARY["max_ring"]
    FAR = MAX_RING + 1
    return CUTOFFS, FAR, MAX_RING, QUARTERS, RUN, SUMMARY, chips, tiles


@app.cell(hide_code=True)
def _(CUTOFFS, MAX_RING, QUARTERS, RUN, mo):
    RULES = {
        "AMW's record up to the year before (the fair test)": "before",
        "AMW's record up to 2023 (strictest)": "fixed:2023",
        f"AMW's whole record, up to {CUTOFFS[-1]} (what a new run would use)": "all",
    }
    rule = mo.ui.dropdown(options=RULES, value=list(RULES)[0], label="History")
    threshold = mo.ui.slider(0.4, 0.95, step=0.01, value=0.4, label="Detection threshold", show_value=True)
    buffer = mo.ui.dropdown(options={"AMW tiles only": 0, **{f"+ {k} ring{'s' if k > 1 else ''}": k for k in range(1, MAX_RING + 1)}},
                            value="+ 1 ring", label="Priority 1 =")
    quarter = mo.ui.dropdown(options=QUARTERS, value=QUARTERS[-1], label="Quarter (map and list)")
    mo.vstack([
        mo.md(f"<small>Run `{RUN.name}`.</small>"),
        mo.hstack([rule, buffer], justify="start", gap=2),
        mo.hstack([threshold, quarter], justify="start", gap=2),
        mo.md("<small>**The fair test** uses only what AMW knew before the year being checked: "
              "for 2025, AMW's record up to 2024; for 2026, up to 2025. Using the whole record "
              "would count 2026 mines that AMW itself found in 2026, which proves nothing. "
              "**Up to 2023** is stricter still: AMW confirms a 2024 mine only when 2025 sees it "
              "again, so the 2024 entries already carry a little 2025 information.</small>"),
    ])
    return buffer, quarter, rule, threshold


@app.cell(hide_code=True)
def _(CUTOFFS, FAR, chips, np, pd, rule, threshold, tiles):
    def cutoff(q):
        """The last year of AMW's record that counts as history for quarter q."""
        kind = rule.value
        if kind == "all":
            return CUTOFFS[-1]
        if kind.startswith("fixed:"):
            return int(kind.split(":")[1])
        return int(q[:4]) - 1

    def ring_of(q):
        """Per tile, its ring around AMW's history for quarter q (FAR: further than we look)."""
        y = cutoff(q)
        col = f"ring_to{min(y, CUTOFFS[-1])}"
        if y < CUTOFFS[0]:
            return pd.Series(FAR, index=tiles["tile"], dtype="int8")
        return tiles.set_index("tile")[col]

    def located(q):
        """Quarter q's chips, each with its tile's ring, and who detected it at the threshold."""
        d = chips[q].assign(tile=chips[q]["chip_id"] // 1000)
        d["ring"] = d["tile"].map(ring_of(q)).fillna(FAR).astype(int)
        d["ours"] = d["confidence"] > threshold.value
        d["amw"] = d["amw_confidence"] > threshold.value     # NaN > x is False
        return d

    view = {q: located(q) for q in chips}
    rings_now = {q: ring_of(q) for q in chips}
    ring_names = {0: "AMW tile", 1: "ring 1", 2: "ring 2", 3: "ring 3"}
    ring_label = lambda r: ring_names.get(r, f"ring {r}") if r < FAR else "further"
    _ = np
    return cutoff, ring_label, rings_now, view


@app.cell(hide_code=True)
def _(FAR, MAX_RING, buffer, cutoff, mo, pd, rings_now, threshold, view):
    def coverage(d, who):
        """Cumulative share of detected chips (and of tiles with a detection) within each ring."""
        det = d[d[who]]
        out = {"detected chips": len(det)}
        for k in range(MAX_RING + 1):
            out[f"within {k}" if k else "in AMW tiles"] = (det["ring"] <= k).mean() if len(det) else float("nan")
        out["missed chips"] = int((det["ring"] > buffer.value).sum())
        tiles_hit = det.groupby("tile")["ring"].first()
        out["tiles with a detection"] = len(tiles_hit)
        out["missed tiles"] = int((tiles_hit > buffer.value).sum())
        return out

    def work(q):
        """Share of all the run's tiles inside each ring: the share of scoring we would do."""
        r = rings_now[q]
        return {("in AMW tiles" if k == 0 else f"within {k}"): (r <= k).mean() for k in range(MAX_RING + 1)}

    _rows, _amw, _work = {}, {}, {}
    for _q, _d in view.items():
        _rows[_q] = {"history to": cutoff(_q)} | coverage(_d, "ours")
        _amw[_q] = {"history to": cutoff(_q)} | coverage(_d, "amw")
        _work[_q] = {"history to": cutoff(_q)} | work(_q)
    ours_table = pd.DataFrame(_rows).T.rename_axis("quarter")
    amw_table = pd.DataFrame(_amw).T.rename_axis("quarter")
    work_table = pd.DataFrame(_work).T.rename_axis("quarter")
    _counts = ["history to", "detected chips", "missed chips", "tiles with a detection", "missed tiles"]
    ours_table = ours_table.astype({c: int for c in _counts})
    amw_table = amw_table.astype({c: int for c in _counts})
    for _t in (ours_table, amw_table, work_table):
        _t["history to"] = _t["history to"].astype(int).astype(str)       # a year, not a number to format

    _k = buffer.value
    _col = "in AMW tiles" if _k == 0 else f"within {_k}"
    _years = sorted({q[:4] for q in view})
    _lines = []
    for _y in _years:
        _qs = [q for q in view if q.startswith(_y)]
        _det = sum(int(ours_table.loc[q, "detected chips"]) for q in _qs)
        _miss = sum(int(ours_table.loc[q, "missed chips"]) for q in _qs)
        _share = work_table.loc[_qs, _col].astype(float).mean()
        _lines.append(f"**{_y}** ({', '.join(q[4:] for q in _qs)}): priority 1 is {_share:.0%} of the tiles and holds "
                      f"{1 - _miss / max(_det, 1):.1%} of the chips we detected; it would miss {_miss:,} of {_det:,}.")
    _ = FAR
    mo.callout(mo.md(f"At threshold {threshold.value:.2f}, priority 1 = {buffer.selected_key.lower()}:\n\n" + "\n\n".join(_lines)),
               kind="info")
    return amw_table, ours_table, work_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How much would priority 1 catch, and how much would it cost?

    **Our detections**: each column is the share of the chips we detected in the quarter that
    lie inside AMW's tiles, or within 1, 2, 3 rings of them. "Missed" counts what lies outside
    the priority-1 area chosen above.

    **AMW's own detections** of the same quarter, the same way: a benchmark that does not
    depend on our pipeline. If AMW's own new detections also sit next to its old ones, the
    pattern is a property of the mines, not of our code.

    **Work**: the share of all the run's tiles inside each area. Priority 1's cost in Earth
    Engine requests and compute is roughly this share of a full run.
    """)
    return


@app.cell(hide_code=True)
def _(amw_table, mo, ours_table, work_table):
    def fmt(t):
        t = t.copy()
        for c in t.columns:
            if c.startswith("within") or c == "in AMW tiles":
                t[c] = t[c].map(lambda v: f"{float(v):.1%}")
        return t.reset_index()

    mo.ui.tabs({
        "our detections": mo.ui.table(fmt(ours_table), selection=None, pagination=False),
        "AMW's detections": mo.ui.table(fmt(amw_table), selection=None, pagination=False),
        "work (share of tiles)": mo.ui.table(fmt(work_table), selection=None, pagination=False),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detections caught against tiles scored

    Each point is one choice of priority 1 (AMW tiles, then one, two, three rings): across,
    the share of tiles we would score; up, the share of our detections it would catch. The
    ideal is top left. A line that keeps climbing steeply means the rings still pay off.
    """)
    return


@app.cell(hide_code=True)
def _(MAX_RING, alt, mo, ours_table, pd, work_table):
    _rows = []
    for _q in ours_table.index:
        for _k in range(MAX_RING + 1):
            _c = "in AMW tiles" if _k == 0 else f"within {_k}"
            _rows.append({"quarter": _q, "area": "AMW tiles" if _k == 0 else f"+{_k} ring{'s' if _k > 1 else ''}",
                          "tiles scored": float(work_table.loc[_q, _c]), "detections caught": float(ours_table.loc[_q, _c])})
        _rows.append({"quarter": _q, "area": "everything", "tiles scored": 1.0, "detections caught": 1.0})
    _d = pd.DataFrame(_rows)
    _base = alt.Chart(_d).encode(
        x=alt.X("tiles scored:Q", title="share of tiles scored", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("detections caught:Q", title="share of our detections caught", axis=alt.Axis(format="%")),
        color=alt.Color("quarter:N"),
        tooltip=["quarter", "area", alt.Tooltip("tiles scored:Q", format=".1%"), alt.Tooltip("detections caught:Q", format=".1%")])
    mo.ui.altair_chart((_base.mark_line(strokeWidth=1.2) + _base.mark_point(filled=True, size=40))
                       .properties(width="container", height=360), chart_selection=False, legend_selection=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the missed detections are

    Every chip we detected in the chosen quarter, coloured by its ring. Light grey dots are
    the centres of the priority-1 tiles. Detections far from any grey are what priority 1
    alone would miss. Scroll to zoom, drag to pan, hover for the chip.
    """)
    return


@app.cell(hide_code=True)
def _(FAR, alt, buffer, mo, pd, quarter, ring_label, rings_now, thin, tiles, view):
    _d = view[quarter.value]
    _d = _d[_d["ours"]].assign(where=lambda x: x["ring"].map(ring_label))
    _order = [ring_label(r) for r in sorted(set(range(FAR)) | {FAR})]
    _colours = ["#8a8984", "#2a78d6", "#e2a400", "#eb6834", "#d64a2a"][:len(_order)]
    _far, _note_far = thin(_d[_d["ring"] > buffer.value], 6_000)
    _near, _note = thin(_d[_d["ring"] <= buffer.value], max(9_000 - len(_far), 2_000))
    _pts = pd.concat([_near, _far])[["chip_id", "subregion", "amw_key_x", "amw_key_y", "confidence", "amw_confidence", "where"]]
    _r = rings_now[quarter.value]
    _p1 = tiles.loc[tiles["tile"].map(_r).le(buffer.value).to_numpy(), ["lon", "lat"]]
    _p1, _note_p1 = thin(_p1, 8_000)
    _bg = alt.Chart(_p1).mark_square(size=6, color="#d9d8d3", opacity=0.6).encode(x="lon:Q", y="lat:Q")
    _fg = (alt.Chart(_pts).mark_circle(size=14)
           .encode(x=alt.X("amw_key_x:Q", title="longitude", scale=alt.Scale(zero=False)),
                   y=alt.Y("amw_key_y:Q", title="latitude", scale=alt.Scale(zero=False)),
                   color=alt.Color("where:N", title=None, sort=_order, scale=alt.Scale(domain=_order, range=_colours)),
                   tooltip=["chip_id", "subregion", "where", alt.Tooltip("confidence:Q", format=".3f"),
                            alt.Tooltip("amw_confidence:Q", format=".3f")]))
    _title = "; ".join(f"{k}: {n}" for k, n in [("inside", _note), ("outside", _note_far), ("tiles", _note_p1)] if n)
    mo.ui.altair_chart((_bg + _fg).properties(width="container", height=520, title=_title).interactive(),
                       chart_selection=False, legend_selection=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Are the missed detections real mines?

    If what priority 1 misses is mostly weak, scattered scores (single chips just above the
    threshold), it is likely noise and the loss is small. If it is clusters of confident
    chips, those are new mining areas away from AMW's record, and priority 2 matters. Raise
    the threshold above to see how the missed share changes with confidence.
    """)
    return


@app.cell(hide_code=True)
def _(buffer, mo, pd, quarter, tiles, view):
    _d = view[quarter.value]
    _det = _d[_d["ours"]].assign(outside=lambda x: x["ring"] > buffer.value)
    _cmp = _det.groupby("outside").agg(chips=("chip_id", "size"), median_confidence=("confidence", "median"),
                                       share_above_075=("confidence", lambda s: (s > 0.75).mean()),
                                       AMW_also_detects=("amw", "mean"))
    _cmp.index = _cmp.index.map({False: "inside priority 1", True: "outside priority 1"})
    _per_tile = (_det[_det["outside"]].groupby("tile")
                 .agg(chips=("chip_id", "size"), max_confidence=("confidence", "max"),
                      AMW_detects=("amw", "sum"), ring=("ring", "first"))
                 .join(tiles.set_index("tile")[["tile_key", "subregion", "lon", "lat"]])
                 .sort_values(["chips", "max_confidence"], ascending=False).reset_index())
    _per_tile["ring"] = _per_tile["ring"].map(lambda r: f"further than {r - 1}" if r > 3 else str(r))
    mo.vstack([
        mo.ui.table(_cmp.round(3).reset_index(names=""), selection=None, pagination=False),
        mo.md(f"**{len(_per_tile):,} tiles outside priority 1** with a detection in {quarter.value}, most chips first. "
              "`AMW_detects`: how many of those chips AMW's own quarter also detects."),
        mo.ui.table(_per_tile.round(4), selection=None, page_size=15),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The priority-1 tile list

    The tiles of the chosen priority-1 area for the chosen quarter's history, with their key
    (the tile's ID in the run) and centre. The squares themselves, for QGIS, are in
    `<run>/amw_comparison/history_tiles.geojson` (AMW's whole record, rings 0-3 as columns).
    """)
    return


@app.cell(hide_code=True)
def _(buffer, cutoff, mo, quarter, rings_now, tiles):
    _r = rings_now[quarter.value]
    _list = tiles.assign(ring=tiles["tile"].map(_r).to_numpy())
    _list = _list[_list["ring"] <= buffer.value][["tile", "tile_key", "subregion", "ring", "history_chips",
                                                  "first_onset_year", "lon", "lat"]]
    _name = f"priority1_history_to{cutoff(quarter.value)}_rings{buffer.value}.csv"
    mo.vstack([
        mo.md(f"{len(_list):,} of {len(tiles):,} tiles (history up to {cutoff(quarter.value)}, {buffer.selected_key.lower()})."),
        mo.download(data=_list.to_csv(index=False).encode(), filename=_name, label=f"Download {_name}"),
    ])
    return


@app.cell(hide_code=True)
def _(SUMMARY, mo, pd):
    _by_year = pd.DataFrame({"AMW locations": SUMMARY["locations_by_onset_year"],
                             "not on our chips": SUMMARY["unmatched_by_onset_year"]}).fillna(0).astype(int)
    mo.vstack([
        mo.md(r"""
        ## What this rests on

        - **AMW's cumulative record** (`amazon_basin_detections.geojson`) holds one row per
          location, with the year mining was first confirmed there (`onset`). A location enters
          it only once two annual composites a year apart both detect it, or, for the latest
          quarters, once one quarter detects it above a stricter threshold. A chip AMW
          detected once in, say, 2020 and never again is *not* in it. So "AMW never found
          mining here" means "never confirmed", and this test is, if anything, too strict.
        - **Location to tile.** AMW's rows carry no tile. Each row's chip is looked up in our
          plan by AMW's own location key, exactly as the chip-by-chip comparison does, and
          our chip ID names its tile. Every AMW location should land on one of our chips;
          on a test region most do not, because they are outside it.
        - **Our detections** are the chips in `chips_<quarter>.parquet` from
          `amw_compare.sbatch` with a quarterly score above the threshold.
        """),
        mo.md(f"File fingerprint `{SUMMARY['cumulative_sha256']}`; {SUMMARY['matched_to_our_chips']:,} of "
              f"{SUMMARY['amw_locations']:,} AMW locations are on our chips; {SUMMARY['tiles_in_run']:,} tiles in the run."),
        mo.ui.table(_by_year.rename_axis("onset year").reset_index(), selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
