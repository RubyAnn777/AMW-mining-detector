"""Small helpers shared by the marimo notebooks (1 to 4, panel_overview, amw_comparison, amw_history,
amw_comparison/compare).

Two things live here:

* `find_runs`: the run folders next to the one given on the command line, for the
  notebooks' run picker.
* `cached`: compute a whole-run table once and keep it in <run>/summaries/, so the
  notebooks open in seconds for everyone who views them. Whoever can write to the run
  folder (the run's owner, or the summaries job) fills the cache; a viewer who can only
  read it gets the stored table, or, if there is none yet, computes it in memory and
  writes nothing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

SUMMARIES = "summaries"

# One colour per frequency, the same in every chart of every notebook.
MONTHLY, QUARTERLY = "#2a78d6", "#eb6834"


def is_run(folder: Path) -> bool:
    return (folder / "run" / "config.yaml").is_file()


def find_runs(default: str | Path | None) -> dict[str, str]:
    """{name: path} of the runs in the folder that holds `default`, newest name last.

    `default` is a run folder (from --run, or $RUN). Its siblings that are runs too are
    offered as well, so a viewer can switch between runs without typing a path.
    """
    if not default:
        return {}
    default = Path(default).expanduser().resolve()
    runs = {default.name: str(default)} if is_run(default) else {}
    try:
        siblings = [p for p in default.parent.iterdir() if p.is_dir() and is_run(p)]
    except OSError:                     # the parent folder is not readable
        siblings = []
    for p in siblings:
        runs.setdefault(p.name, str(p))
    return dict(sorted(runs.items()))


def newest_change(paths: Iterable[Path]) -> float:
    """The latest modification time among `paths` (0 if none exists)."""
    times = []
    for p in paths:
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(times, default=0.0)


def is_cached(run: Path, name: str, inputs: Iterable[Path]) -> bool:
    """True if `cached` would return a stored result for `name` without computing."""
    path = Path(run) / SUMMARIES / f"{name}.pkl"
    return path.is_file() and path.stat().st_mtime >= newest_change(inputs)


def can_write(run: Path) -> bool:
    """True if this user can store summaries in the run (its owner can; a viewer can't)."""
    folder = Path(run) / SUMMARIES
    return os.access(folder if folder.is_dir() else Path(run), os.W_OK)


def cached(run: Path, name: str, compute: Callable[[], object], inputs: Iterable[Path]) -> object:
    """compute(), stored as <run>/summaries/<name>.pkl and reused while it is up to date.

    The stored result is reused when it is newer than every file in `inputs` (for example
    the panel's _record.json files: a newly assembled period makes it stale). A user
    without write access to the run gets the result without anything being written.
    """
    path = Path(run) / SUMMARIES / f"{name}.pkl"
    inputs = list(inputs)
    try:
        if path.is_file() and path.stat().st_mtime >= newest_change(inputs):
            return pd.read_pickle(path)
    except Exception:                   # unreadable or from an incompatible pandas: recompute
        pass
    value = compute()
    tmp = path.with_name(f"_{path.name}.{os.getpid()}")
    try:
        path.parent.mkdir(exist_ok=True)
        pd.to_pickle(value, tmp)
        os.replace(tmp, path)           # appears complete or not at all
    except OSError:                     # read-only for this user: keep it in memory only
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return value


# A chart sends every row it draws to the browser, and marimo refuses outputs above a few
# megabytes (8 MB). A full run has hundreds of thousands of chips per quarter, so point charts
# draw a random sample of at most this many rows and say so in their title.
MAX_POINTS = 10_000


def thin(df: pd.DataFrame, n: int = MAX_POINTS, every: str | None = None) -> tuple[pd.DataFrame, str]:
    """At most `n` rows of `df` for a chart, and a note saying how many were left out.

    Without `every`, a random sample (the same one each time). With `every`, a column name:
    rows are kept at even steps within each value of that column, in their order, so a
    line per task stays a line.
    """
    if len(df) <= n:
        return df, ""
    if every is None:
        out = df.sample(n, random_state=0)
    else:
        step = -(-len(df) // n)                                   # ceiling division
        out = df[df.groupby(every).cumcount() % step == 0]
    return out, f"showing {len(out):,} of {len(df):,}, picked at random" if every is None else \
        f"showing one in {step} of {len(df):,} points"


# What a section that reads the whole run shows in the browser before its summary is stored.
NOT_STORED = ("*Not stored yet. This section reads the whole run, which takes too long to do while "
              "you wait; Ruby stores it with `cluster/summaries.sbatch`.*")


def in_browser() -> bool:
    """True when the notebook is open as an app in a browser (view.sbatch, `marimo run`)."""
    try:
        import marimo as mo
        return mo.app_meta().mode == "run"
    except Exception:                   # not inside marimo
        return False


def stored(run: Path, name: str, compute: Callable[[], object], inputs: Iterable[Path]) -> object | None:
    """Like `cached`, but in the browser only a stored result, or None if there is none.

    For summaries that read every period of a run: on a full run they take many minutes,
    and a page that computes them while it opens looks frozen. summaries.sbatch (which runs
    the notebooks as scripts) and the editor compute and store them.
    """
    inputs = list(inputs)
    if in_browser() and not is_cached(run, name, inputs):
        return None
    return cached(run, name, compute, inputs)


# ------------------------------------------------------------------ figures and their captions
# Every figure in the analysis notebooks is followed by a caption that says what one mark is,
# what each axis and colour measures, and which chips and periods are in it.

# Colour-blind-safe colours (Okabe and Ito), used with one meaning each across the notebooks.
DETECTED, MISSED, MISSED_CLOUDY, NOT_SCORED = "#0072B2", "#E69F00", "#999999", "#DDDDDD"
OURS, AMW = "#0072B2", "#D55E00"
YEARLY = "#009E73"
SEQUENTIAL = "viridis"           # maps and heat maps: one ordered scale, readable in grey too


def caption(number: int | str, title: str, shows: str, *, mark: str | None = None, x: str | None = None,
            y: str | None = None, colour: str | None = None, sample: str | None = None,
            note: str | None = None):
    """The caption under a figure, as a marimo element. Write `shows` as what the figure ACTUALLY
    shows (not what it is meant to suggest); the keywords name each encoding."""
    import marimo as mo
    lines = [f"**Figure {number}. {title}** {shows}"]
    for label, text in (("One mark", mark), ("Horizontal axis", x), ("Vertical axis", y),
                        ("Colour", colour), ("Chips and periods", sample), ("Note", note)):
        if text:
            lines.append(f"*{label}:* {text}")
    html = mo.md("  \n".join(lines)).text
    return mo.Html(f'<div style="font-size:0.92rem;line-height:1.45;color:var(--muted-foreground,#555);'
                   f'border-left:3px solid #ccc;padding:2px 0 2px 10px;margin:4px 0 28px 0">{html}</div>')


def figure(chart, number: int | str, title: str, shows: str, **parts):
    """A chart with its caption underneath (see `caption` for the parts)."""
    import marimo as mo
    return mo.vstack([chart, caption(number, title, shows, **parts)])


def altair_theme():
    """Larger, plainer charts than altair's default; call once per notebook."""
    import altair as alt

    @alt.theme.register("amw", enable=True)
    def _():
        return alt.theme.ThemeConfig({
            "config": {
                "font": "Helvetica Neue, Arial, sans-serif",
                "axis": {"labelFontSize": 12, "titleFontSize": 13, "titleFontWeight": "normal",
                         "gridColor": "#eee", "domainColor": "#888", "tickColor": "#888"},
                "legend": {"labelFontSize": 12, "titleFontSize": 13, "titleFontWeight": "normal", "titleLimit": 500,
                           "labelLimit": 400,
                           "orient": "bottom", "direction": "horizontal"},
                "axisXBand": {"labelAngle": -40},
                "header": {"labelFontSize": 13, "titleFontSize": 13},
                "title": {"fontSize": 14, "fontWeight": "normal", "anchor": "start"},
                "view": {"stroke": None},
            }})
    alt.data_transformers.disable_max_rows()


def glossary(names: list[str], intro: str = ""):
    """The variables a notebook uses, one readable paragraph each, inside a fold-out section.

    Not a table: a table squeezes long definitions into narrow cells. Definitions come from
    panel_pipeline/lib/variables.py, the one place they are written."""
    import marimo as mo
    from variables import describe
    blocks = [intro] if intro else []
    for v in describe(names):
        where = f" · stored in: {v['where']}" if v.get("where") and v["where"] != "derived" else (
            " · computed in the notebooks" if v.get("where") == "derived" else "")
        block = f"**`{v['variable']}`** · {v['unit']}{where}  \n{v['meaning']}"
        if v.get("missing") and v["missing"] != "never":
            block += f"  \n*Missing when:* {v['missing']}."
        blocks.append(block)
    return mo.accordion({f"The variables in this notebook ({len(names)}), and what each one means":
                         mo.md("\n\n".join(blocks))})


def period_map(cells, tags: list[str], layers: list[dict], height: int = 560):
    """Up to two maps side by side, one value per 0.25 degree cell, with a time slider and play button.

    cells: rows of (period, cell_lon, cell_lat, <value columns>), as analysis_tables.to_cells(..., by=["period"]).
    layers: [{"column", "title", "colorscale", "reverse" (bool)}], one or two. The slider runs in the browser,
    so it also works in a static HTML export. The first frame carries the cell outlines and the
    others only the values, which keeps an exported file small.
    """
    import numpy as np
    import plotly.graph_objects as go

    cells = cells.copy()
    cells["period"] = cells["period"].astype(str)
    cells["id"] = cells["cell_lon"].round(4).astype(str) + "," + cells["cell_lat"].round(4).astype(str)
    ids = cells.drop_duplicates("id")
    half = float(np.round(np.diff(np.unique(ids["cell_lon"])).min(), 4) / 2) if ids["cell_lon"].nunique() > 1 else 0.125
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": r.id, "properties": {},
         "geometry": {"type": "Polygon", "coordinates": [[[r.cell_lon - half, r.cell_lat - half], [r.cell_lon + half, r.cell_lat - half],
                                                          [r.cell_lon + half, r.cell_lat + half], [r.cell_lon - half, r.cell_lat + half],
                                                          [r.cell_lon - half, r.cell_lat - half]]]}}
        for r in ids.itertuples()]}
    all_ids = ids["id"].tolist()
    n = len(layers)
    width = 0.45 if n == 2 else 0.92
    lims = {}
    for L in layers:
        v = cells[L["column"]].astype(float)
        lims[L["column"]] = (float(np.nanquantile(v, 0.02)) if L.get("from_low") else 0.0,
                             float(np.nanquantile(v, 0.98)) if v.notna().any() else 1.0)

    def traces(tag, full):
        c = cells[cells["period"] == tag].set_index("id")
        out = []
        for i, L in enumerate(layers):
            z = c[L["column"]].reindex(all_ids).astype(float).round(4).tolist()
            extra = dict(geojson=geo, locations=all_ids) if full else {}
            lo, hi = lims[L["column"]]
            out.append(go.Choroplethmap(z=z, zmin=lo, zmax=hi, colorscale=L.get("colorscale", "Viridis"),
                                        reversescale=L.get("reverse", False), marker_line_width=0,
                                        subplot="map" if i == 0 else "map2",
                                        hovertemplate=L["title"] + ": %{z:.3f}<extra></extra>",
                                        colorbar=dict(title=dict(text=L["title"], side="right"),
                                                      x=(width + 0.01) if i == 0 else 1.0, len=0.8), **extra))
        return out

    centre = dict(lon=float(cells["cell_lon"].mean()), lat=float(cells["cell_lat"].mean()))
    span = float(max(cells["cell_lon"].max() - cells["cell_lon"].min(), cells["cell_lat"].max() - cells["cell_lat"].min()))
    zoom = 3.2 if span > 10 else 4.5 if span > 3 else 7
    fig = go.Figure(data=traces(tags[-1], True) if tags else [],
                    frames=[go.Frame(data=traces(t, False), name=t, traces=list(range(n))) for t in tags])
    lay = dict(map=dict(style="carto-positron", center=centre, zoom=zoom, domain=dict(x=[0, width], y=[0, 1])))
    if n == 2:
        lay["map2"] = dict(style="carto-positron", center=centre, zoom=zoom, domain=dict(x=[0.53, 0.98], y=[0, 1]))
    fig.update_layout(
        **lay, height=height, margin=dict(l=0, r=0, t=10, b=0),
        sliders=[dict(active=len(tags) - 1, currentvalue=dict(prefix="period: "), pad=dict(t=40), x=0.1, len=0.9,
                      steps=[dict(label=t, method="animate", args=[[t], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in tags])],
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.06, xanchor="left", yanchor="top",
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True)])])])
    return fig
