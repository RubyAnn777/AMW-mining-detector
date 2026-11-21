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
    import notebook_kit as kit
    from notebook_kit import thin

    COUNTRIES = REPO / "data" / "boundaries" / "south_america_countries.geojson"
    kit.altair_theme()
    return COUNTRIES, Path, alt, json, kit, mo, np, os, pd, thin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Could we score the tiles where AMW has detected mining first?

    **The plan.** Priority 1: every tile in which AMW has ever detected mining (any chip in
    any of AMW's raw detection files, confidence above 0.4, every year and quarter we have),
    plus a few rings of neighbouring tiles. Priority 2: the rest of the basin, later. In the
    end the whole basin is scored, so priority 1 decides only what we have first.

    **The check.** For 2025 and 2026 we have scored the whole basin already. Of the chips we
    detected in each quarter, how many lie in priority 1? Whatever lies outside is what we
    would only see after priority 2.

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
    SUMMARY = json.loads((RESULTS / "history_summary.json").read_text())
    mo.stop("files" not in SUMMARY, mo.callout(mo.md(
        "These history tiles were made from AMW's cumulative record by the earlier version. "
        "Run `sbatch cluster/amw_history.sbatch $RUN` again to rebuild them from the raw files."), kind="warn"))
    tiles = pd.read_parquet(RESULTS / "history_tiles.parquet")
    _available = sorted(p.stem.removeprefix("chips_") for p in RESULTS.glob("chips_*.parquet"))
    QUARTERS = [q for q in _available if q[:4] >= "2025"]
    mo.stop(not QUARTERS, mo.callout(mo.md(
        f"No compared quarters from 2025 on in `{RESULTS}`. They come from "
        f"`sbatch cluster/amw_compare.sbatch $RUN 2025` (and 2026)."), kind="warn"))
    chips = {q: pd.read_parquet(RESULTS / f"chips_{q}.parquet",
                                columns=["chip_id", "subregion", "amw_key_x", "amw_key_y", "confidence"])
             for q in QUARTERS}
    MAX_RING = SUMMARY["max_ring"]
    FAR = MAX_RING + 1
    return FAR, MAX_RING, QUARTERS, RUN, SUMMARY, chips, tiles


@app.cell(hide_code=True)
def _(MAX_RING, QUARTERS, RUN, SUMMARY, mo):
    threshold = mo.ui.slider(0.4, 0.95, step=0.01, value=0.4, label="Detection threshold", show_value=True)
    buffer = mo.ui.dropdown(options={"AMW tiles only": 0, **{f"AMW tiles + {k} ring{'s' if k > 1 else ''}": k
                                                             for k in range(1, MAX_RING + 1)}},
                            value=f"AMW tiles + {MAX_RING} rings" if MAX_RING > 1 else "AMW tiles only", label="Priority 1 =")
    quarter = mo.ui.dropdown(options=["all quarters"] + QUARTERS, value="all quarters", label="Map and lists")
    mo.vstack([
        mo.md(f"<small>Run `{RUN.name}`. AMW history from {len(SUMMARY['files'])} raw files: "
              f"{', '.join(SUMMARY['files'])}.</small>"),
        mo.hstack([buffer, threshold, quarter], justify="start", gap=2),
    ])
    return buffer, quarter, threshold


@app.cell(hide_code=True)
def _(kit):
    kit.glossary(["confidence", "ring", "priority 1", "subregion"])
    return


@app.cell(hide_code=True)
def _(FAR, chips, threshold, tiles):
    _ring = tiles.set_index("tile")["ring"]
    _country = tiles.set_index("tile")["country"]

    def located(q):
        """Quarter q's chips we detected at the threshold, each with its tile's ring and country."""
        d = chips[q]
        d = d[d["confidence"] > threshold.value].assign(tile=lambda x: x["chip_id"] // 1000, quarter=q)
        d["ring"] = d["tile"].map(_ring).fillna(FAR).astype(int)
        d["country"] = d["tile"].map(_country).fillna("")
        return d

    view = {q: located(q) for q in chips}
    ring_label = lambda r: ("AMW tile" if r == 0 else f"ring {r}") if r < FAR else "further"
    return ring_label, view


@app.cell(hide_code=True)
def _(MAX_RING, buffer, mo, pd, threshold, tiles, view):
    def coverage(det):
        """Cumulative share of detected chips within each ring, and what priority 1 misses."""
        out = {"detected chips": len(det)}
        for k in range(MAX_RING + 1):
            out["in AMW tiles" if k == 0 else f"within {k}"] = (det["ring"] <= k).mean() if len(det) else float("nan")
        out["missed chips"] = int((det["ring"] > buffer.value).sum())
        hit = det.groupby("tile")["ring"].first()
        out["tiles with a detection"] = len(hit)
        out["missed tiles"] = int((hit > buffer.value).sum())
        return out

    _work = {("in AMW tiles" if k == 0 else f"within {k}"): (tiles["ring"] <= k).mean() for k in range(MAX_RING + 1)}
    _rows = {q: coverage(d) for q, d in view.items()}
    for _y in sorted({q[:4] for q in view}):
        _rows[f"{_y} (all quarters)"] = coverage(pd.concat([d for q, d in view.items() if q.startswith(_y)], ignore_index=True))
    coverage_table = pd.DataFrame(_rows).T.rename_axis("quarter")
    _counts = ["detected chips", "missed chips", "tiles with a detection", "missed tiles"]
    coverage_table = coverage_table.astype({c: int for c in _counts})
    work = pd.Series(_work)

    _col = "in AMW tiles" if buffer.value == 0 else f"within {buffer.value}"
    _lines = [f"**{r}**: holds {coverage_table.loc[r, _col]:.1%} of the chips we detected; "
              f"misses {coverage_table.loc[r, 'missed chips']:,} of {coverage_table.loc[r, 'detected chips']:,} "
              f"(in {coverage_table.loc[r, 'missed tiles']:,} tiles)."
              for r in coverage_table.index if "all quarters" in r]
    mo.callout(mo.md(
        f"**Priority 1 = {buffer.selected_key}: {(tiles['ring'] <= buffer.value).sum():,} of {len(tiles):,} "
        f"tiles ({work[_col]:.1%} of the basin).** At threshold {threshold.value:.2f}:\n\n" + "\n\n".join(_lines)),
        kind="info")
    return coverage_table, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What priority 1 holds, quarter by quarter

    Each "within" column is the share of the chips we detected that lie inside AMW's tiles,
    or within 1, 2, 3 rings of them. "Missed" counts what lies outside the priority-1 area
    chosen above. The last row is the cost: the share of all the run's tiles in each area,
    which is roughly the share of Earth Engine requests and compute of a full run.
    """)
    return


@app.cell(hide_code=True)
def _(coverage_table, mo, pd, work):
    _t = coverage_table.copy()
    _t.loc["tiles to score (cost)", work.index] = work.to_numpy()
    for _c in work.index:
        _t[_c] = _t[_c].map(lambda v: f"{float(v):.1%}")
    _t = _t.astype({c: "Int64" for c in ["detected chips", "missed chips", "tiles with a detection", "missed tiles"]})
    _ = pd
    mo.accordion({"The coverage table (the numbers behind Figure 1)": mo.ui.table(_t.reset_index(), selection=None, pagination=False)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detections caught against tiles scored

    Each point is one choice of priority 1 (AMW tiles, then one, two, three rings): across,
    the share of tiles we would score; up, the share of our detections it would catch. Top
    left is ideal; where the line flattens, more rings stop paying off.
    """)
    return


@app.cell(hide_code=True)
def _(MAX_RING, alt, coverage_table, kit, mo, pd, work):
    _rows = []
    for _q in coverage_table.index:
        for _k in range(MAX_RING + 1):
            _c = "in AMW tiles" if _k == 0 else f"within {_k}"
            _rows.append({"quarter": _q, "area": "AMW tiles" if _k == 0 else f"+{_k} ring{'s' if _k > 1 else ''}",
                          "tiles scored": float(work[_c]), "detections caught": float(coverage_table.loc[_q, _c])})
    _d = pd.DataFrame(_rows)
    _base = alt.Chart(_d).encode(
        x=alt.X("tiles scored:Q", title="share of tiles scored", axis=alt.Axis(format="%"), scale=alt.Scale(zero=True)),
        y=alt.Y("detections caught:Q", title="share of our detections caught", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False)),
        color=alt.Color("quarter:N"),
        tooltip=["quarter", "area", alt.Tooltip("tiles scored:Q", format=".1%"), alt.Tooltip("detections caught:Q", format=".1%")])
    mo.vstack([
        mo.ui.altair_chart((_base.mark_line(strokeWidth=1.2) + _base.mark_point(filled=True, size=40))
                           .properties(width="container", height=340), chart_selection=False, legend_selection=True),
        kit.caption(1, "Detections caught against tiles scored.",
                    "For each choice of priority 1 (AMW tiles only, then with one, two, three rings around them), the share "
                    "of the run's tiles it contains and the share of our detected chips that lie in it. One line per quarter, "
                    "and one per year with all its quarters pooled.",
                    mark="one choice of priority 1 for one quarter or year.", x="share of all the run's tiles in priority 1 "
                    "(roughly the share of compute).", y="share of our detected chips inside priority 1.",
                    colour="quarter or year; click a legend entry to show only that one.",
                    sample="chips we detected above the chosen threshold, from the compared quarters of 2025 on."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By country

    Our detections per country, and how many priority 1 would miss there. Countries are
    assigned by tile centre on coarse outlines (Natural Earth 1:110m), so a tile right on a
    border can land in the neighbour.
    """)
    return


@app.cell(hide_code=True)
def _(alt, buffer, kit, mo, pd, quarter, view):
    _d = pd.concat(view.values(), ignore_index=True) if quarter.value == "all quarters" else view[quarter.value]
    _d = _d.assign(missed=_d["ring"] > buffer.value)
    _d["missed_tile"] = _d["tile"].where(_d["missed"])
    _by = (_d.groupby("country")
           .agg(detected_chips=("chip_id", "size"), missed_chips=("missed", "sum"),
                tiles=("tile", "nunique"), missed_tiles=("missed_tile", "nunique"))
           .sort_values("detected_chips", ascending=False))
    _by["share missed"] = (_by["missed_chips"] / _by["detected_chips"]).map(lambda v: f"{v:.1%}")
    _long = _by.reset_index().assign(inside=lambda x: x["detected_chips"] - x["missed_chips"])
    _long = _long.melt(id_vars="country", value_vars=["inside", "missed_chips"], var_name="where", value_name="chips")
    _long["where"] = _long["where"].map({"inside": "inside priority 1", "missed_chips": "outside (missed)"})
    _chart = alt.Chart(_long).mark_bar().encode(
        y=alt.Y("country:N", sort=list(_by.index), title=None), x=alt.X("chips:Q", title="detected chips"),
        color=alt.Color("where:N", scale=alt.Scale(domain=["inside priority 1", "outside (missed)"], range=["#8a8984", kit.AMW]),
                        title=None),
        tooltip=["country", "where", alt.Tooltip("chips:Q", format=",")]).properties(width="container", height=26 * len(_by))
    mo.vstack([kit.figure(_chart, 2, f"Our detections per country, {quarter.value}.",
                          "Chips we detected in each country, split into those inside the chosen priority-1 area and those "
                          "outside it, which priority 1 would miss.",
                          mark="one country; bar length = detected chips.", x="detected chips.",
                          colour="grey: inside priority 1; red: outside.",
                          sample=f"chips detected above the chosen threshold, {quarter.value}."),
               mo.accordion({"The numbers (table)": mo.ui.table(_by.reset_index(), selection=None, pagination=False)})])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Map

    Every chip we detected, coloured by its ring; grey squares are the priority-1 tiles;
    black lines are country borders, to compare with the Amazon Mining Watch website.
    Detections outside the grey are what priority 1 misses. Scroll to zoom, drag to pan,
    hover for the chip; click a legend entry to show only that group.
    """)
    return


@app.cell(hide_code=True)
def _(COUNTRIES, FAR, alt, buffer, kit, mo, pd, quarter, ring_label, thin, tiles, view):
    import geopandas as _gpd
    import shapely as _shp

    _d = pd.concat(view.values(), ignore_index=True) if quarter.value == "all quarters" else view[quarter.value]
    _d = _d.assign(where=_d["ring"].map(ring_label))
    _order = [ring_label(r) for r in range(FAR + 1)]
    _colours = ["#8a8984", "#2a78d6", "#5bb5a2", "#e2a400", "#d64a2a"][:len(_order)]
    _far, _note_far = thin(_d[_d["ring"] > buffer.value], 6_000)
    _near, _note = thin(_d[_d["ring"] <= buffer.value], max(9_000 - len(_far), 2_000))
    _pts = pd.concat([_near, _far])[["chip_id", "quarter", "country", "amw_key_x", "amw_key_y", "confidence", "where"]]
    _p1, _note_p1 = thin(tiles.loc[tiles["ring"] <= buffer.value, ["lon", "lat"]], 12_000)

    _x0, _x1 = min(_pts["amw_key_x"].min(), _p1["lon"].min()) - 0.5, max(_pts["amw_key_x"].max(), _p1["lon"].max()) + 0.5
    _y0, _y1 = min(_pts["amw_key_y"].min(), _p1["lat"].min()) - 0.5, max(_pts["amw_key_y"].max(), _p1["lat"].max()) + 0.5
    _xs, _ys = alt.Scale(domain=[_x0, _x1], zero=False), alt.Scale(domain=[_y0, _y1], zero=False)

    _layers = []
    if COUNTRIES.exists():
        _c = _gpd.read_file(COUNTRIES)
        _lines = []
        for _name, _geom in zip(_c["country"], _c.geometry):
            for _i, _ring in enumerate(_shp.get_parts(_shp.boundary(_geom))):
                _xy = _shp.get_coordinates(_ring)
                _lines.append(pd.DataFrame({"lon": _xy[:, 0], "lat": _xy[:, 1], "seg": f"{_name}{_i}",
                                            "order": range(len(_xy))}))
        _b = pd.concat(_lines)
        _layers.append(alt.Chart(_b).mark_line(color="black", strokeWidth=0.7, clip=True)
                       .encode(x=alt.X("lon:Q", scale=_xs), y=alt.Y("lat:Q", scale=_ys), detail="seg:N", order="order:Q"))
        _lab = _c.assign(lon=_shp.get_x(_c.geometry.representative_point()), lat=_shp.get_y(_c.geometry.representative_point()))
        _lab = _lab[_lab["lon"].between(_x0, _x1) & _lab["lat"].between(_y0, _y1)]
        _layers.append(alt.Chart(_lab[["country", "lon", "lat"]]).mark_text(fontSize=12, fontWeight="bold", color="#444")
                       .encode(x=alt.X("lon:Q", scale=_xs), y=alt.Y("lat:Q", scale=_ys), text="country:N"))
    _bg = (alt.Chart(_p1).mark_square(size=10, color="#d9d8d3", opacity=0.7)
           .encode(x=alt.X("lon:Q", scale=_xs), y=alt.Y("lat:Q", scale=_ys)))
    _fg = (alt.Chart(_pts).mark_circle(size=14)
           .encode(x=alt.X("amw_key_x:Q", title="longitude", scale=_xs),
                   y=alt.Y("amw_key_y:Q", title="latitude", scale=_ys),
                   color=alt.Color("where:N", title=None, sort=_order, scale=alt.Scale(domain=_order, range=_colours)),
                   tooltip=["chip_id", "quarter", "country", "where", alt.Tooltip("confidence:Q", format=".3f")]))
    _title = "; ".join(f"{k}: {n}" for k, n in [("inside", _note), ("outside", _note_far), ("tiles", _note_p1)] if n)
    mo.vstack([
        mo.ui.altair_chart(alt.layer(_bg, *_layers, _fg).properties(width="container", height=600, title=_title).interactive(),
                           chart_selection=False, legend_selection=True),
        kit.caption(3, f"Our detections and the priority-1 tiles, {quarter.value}.",
                    "Every chip we detected, at its position and coloured by the ring of its tile, over the priority-1 tiles "
                    "(grey squares) and country borders (black lines). Coloured points outside the grey are detections "
                    "priority 1 would miss.",
                    mark="one detected chip; grey square: one priority-1 tile.", x="longitude.", y="latitude.",
                    colour="ring of the chip's tile: grey = AMW tile, then ring 1, 2, ..., and 'further'.",
                    sample="chips detected above the threshold; when there are many, points are thinned (the plot title "
                           "says by how much) and the missed ones are kept first."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The missed tiles

    Tiles outside priority 1 with a detection, most chips first. A cluster of confident
    chips is likely a new mining area away from anything AMW has seen; a single chip just
    above the threshold is more likely noise.
    """)
    return


@app.cell(hide_code=True)
def _(buffer, mo, pd, quarter, tiles, view):
    _d = pd.concat(view.values(), ignore_index=True) if quarter.value == "all quarters" else view[quarter.value]
    _out = _d[_d["ring"] > buffer.value]
    _per_tile = (_out.groupby("tile")
                 .agg(chips=("chip_id", "nunique"), quarters=("quarter", lambda q: " ".join(sorted(set(q)))),
                      max_confidence=("confidence", "max"))
                 .join(tiles.set_index("tile")[["tile_key", "country", "subregion", "lon", "lat"]])
                 .sort_values(["chips", "max_confidence"], ascending=False).reset_index())
    _per_tile["tile"] = _per_tile["tile"].astype(str)
    mo.vstack([
        mo.md(f"**{len(_per_tile):,} tiles** outside priority 1 with a detection ({quarter.value})."),
        mo.ui.table(_per_tile.round(4), selection=None, page_size=15),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The priority-1 tile list

    The tiles of the chosen priority-1 area with their key (the tile's ID in the run), country
    and centre. The squares, for QGIS, are in `<run>/amw_comparison/history_tiles.geojson`, and
    the chip IDs of every location AMW ever detected in `history_chips.parquet`.
    """)
    return


@app.cell(hide_code=True)
def _(buffer, mo, tiles):
    _list = tiles[tiles["ring"] <= buffer.value][["tile", "tile_key", "country", "subregion", "ring",
                                                  "history_chips", "first_year", "lon", "lat"]]
    _name = f"priority1_rings{buffer.value}.csv"
    mo.vstack([
        mo.md(f"{len(_list):,} of {len(tiles):,} tiles ({buffer.selected_key})."),
        mo.download(data=_list.to_csv(index=False).encode(), filename=_name, label=f"Download {_name}"),
    ])
    return


@app.cell(hide_code=True)
def _(SUMMARY, mo, pd):
    _files = pd.DataFrame({"file": SUMMARY["files"], "fingerprint": SUMMARY["fingerprints"]}).rename_axis("period")
    _years = pd.Series(SUMMARY["locations_by_first_year"], name="locations first detected").rename_axis("year")
    mo.vstack([
        mo.md(r"""
        ## What this rests on

        - **AMW's history** is every location in any of AMW's raw detection files
          (`amazon_basin_<period>_t0.4.geojson`): the model's direct output above 0.4, before
          postprocessing. Not AMW's cumulative record, which is cleaned (isolated chips need
          0.75, a mine needs two years to be confirmed), and not the Andes files (threshold 0.2).
          The 2025 and 2026 files are included, since the whole basin is scored in the end anyway.
        - **Location to tile.** AMW's rows carry no chip ID. Each row's chip is looked up in our
          plan by AMW's own location key, as the chip-by-chip comparison does, and our chip ID
          names its tile.
        - **Our detections** are the chips in `chips_<quarter>.parquet` from `amw_compare.sbatch`
          with a quarterly score above the threshold.
        """),
        mo.md(f"{SUMMARY['matched_to_our_chips']:,} of {SUMMARY['amw_locations']:,} AMW locations are on our chips "
              f"({SUMMARY['history_chips']:,} chip IDs, counting both copies where tiles overlap); "
              f"{SUMMARY['tiles_in_run']:,} tiles in the run."),
        mo.hstack([mo.ui.table(_files.reset_index(), selection=None, pagination=False),
                   mo.ui.table(_years.reset_index(), selection=None, pagination=False)], align="start"),
    ])
    return


if __name__ == "__main__":
    app.run()
