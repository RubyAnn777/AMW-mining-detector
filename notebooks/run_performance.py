import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Run performance")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "panel_pipeline" / "checks"))
    sys.path.insert(0, str(REPO / "panel_pipeline" / "lib"))
    import performance as perf
    from config import load_config
    from notebook_kit import find_runs

    _ = alt.data_transformers.disable_max_rows()
    return Path, alt, find_runs, load_config, mo, os, pd, perf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run performance

    How fast the scoring went and what held it back. The notebook reads the progress
    lines `detect.py` writes into each task folder (`_log.txt`, `_status.json`) and Slurm's
    job accounting. All of these are small text files, so this opens quickly even for a
    full run.
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
def _(Path, mo, perf, run_picker):
    RUN = Path(run_picker.value or ".")
    mo.stop(not (RUN / "detect").is_dir(), mo.callout(mo.md(f"No `detect/` folder in `{RUN}`. Pick a run above."), kind="warn"))
    lines = perf.progress(RUN)
    _sessions = perf.sessions(RUN, lines)
    jobs, _note = perf.find_jobs(RUN, _sessions, None, "Europe/Berlin")
    sessions = perf.assign_jobs(_sessions, jobs)
    tasks = perf.task_summary(sessions)
    mo.md(f"`{RUN}`: {len(lines):,} progress lines from {len(sessions)} sessions of {len(tasks)} tasks. {_note}.")
    return RUN, jobs, lines, sessions, tasks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tasks

    One row per task. `tiles_per_min` is tiles saved over the hours its sessions ran, slow
    start and tail included; `rate` is the median of its progress lines, the speed while it
    was running. `scoring_busy` near 100 % means the node's cores were the limit;
    `workers_busy` near 100 % with scoring well below means the downloads from Earth Engine
    were. Click a column to sort, or filter it.
    """)
    return


@app.cell(hide_code=True)
def _(mo, tasks):
    _columns = ["frequency", "period", "subregion", "tiles_saved", "tiles_planned", "failed", "complete",
                "sessions", "hours", "tiles_per_min", "rate", "dl_median_s", "workers_busy", "scoring_busy",
                "retries", "workers", "limit", "nodes", "limited_by"]
    mo.ui.table(tasks[[c for c in _columns if c in tasks]].round(2), selection=None, page_size=15) if len(tasks) \
        else mo.md("No task has written progress lines yet.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Speed over time

    One line per task. A task starts slowly while it loads the model. A task that stays slow
    with long downloads is usually on a slow node: on 28 Oct 2026 node22-006 needed 18 to 35 s
    per download where the other nodes needed 5 s. The drop at the end of a node's work is the
    tail, when the last tasks of a job run alone and the rest of the node idles.
    """)
    return


@app.cell(hide_code=True)
def _(lines, mo):
    busy = lines[lines["rate"] > 0] if len(lines) else lines
    measure = mo.ui.radio(options={"tiles per minute": "rate", "median download time (s)": "dl_median_s"},
                          value="tiles per minute", label="Show", inline=True)
    task_filter = mo.ui.multiselect(options=sorted(busy["task"].unique()) if len(busy) else [], label="Tasks (empty = all)")
    mo.hstack([measure, task_filter], justify="start", gap=2)
    return busy, measure, task_filter


@app.cell(hide_code=True)
def _(alt, busy, measure, mo, task_filter):
    mo.stop(busy.empty, mo.md("No progress lines: the `_log.txt` files are missing from this copy of the run."))
    _d = busy[busy["task"].isin(task_filter.value)] if task_filter.value else busy
    _title = [k for k, v in measure.options.items() if v == measure.value][0]
    mo.ui.altair_chart(
        alt.Chart(_d).mark_line(point=alt.OverlayMarkDef(size=12), strokeWidth=1)
        .encode(x=alt.X("time:T", title="UTC"), y=alt.Y(f"{measure.value}:Q", title=_title),
                color=alt.Color("task:N", legend=alt.Legend(columns=2, symbolLimit=40)),
                tooltip=["task", "time:T", alt.Tooltip(f"{measure.value}:Q", format=".1f")])
        .properties(height=320, width="container").interactive(bind_y=False),
        chart_selection=False, legend_selection=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What was the limit

    Each point is one progress line. Top left: the node's scoring was the limit, and more
    download workers would not have helped. Bottom right: the downloads were. Top right: both
    were busy, which is where a well-sized job sits.
    """)
    return


@app.cell(hide_code=True)
def _(alt, busy, mo, tasks):
    mo.stop(busy.empty)
    _points = alt.Chart(busy).mark_circle(size=14, opacity=0.5).encode(
        x=alt.X("workers_busy:Q", title="download workers busy, %", scale=alt.Scale(domain=[0, 105])),
        y=alt.Y("scoring_busy:Q", title="scoring busy, %", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color("task:N", legend=None), tooltip=["task", "time:T", "workers_busy", "scoring_busy"])
    _rules = (alt.Chart().mark_rule(strokeDash=[3, 3], color="grey").encode(x=alt.datum(80))
              + alt.Chart().mark_rule(strokeDash=[3, 3], color="grey").encode(y=alt.datum(80)))
    mo.hstack([(_points + _rules).properties(width=360, height=360),
               tasks["limited_by"].value_counts().rename_axis("limited by").reset_index()],
              justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Jobs and nodes

    Each session is matched to the Slurm job that was running when it started. A job's
    `tiles_per_core_hour` is the figure to plan with; a node far below the others at the same
    settings is a slow node and belongs in the `--exclude` list.
    """)
    return


@app.cell(hide_code=True)
def _(jobs, mo, perf, sessions, settings):
    job_table = perf.job_summary(sessions, jobs, settings.value["cores"])
    if len(job_table):
        _by_node = job_table[job_table["tiles"] > 0].groupby("node")[["hours", "core_hours", "tiles"]].sum()
        _by_node["tiles_per_core_hour"] = _by_node["tiles"] / _by_node["core_hours"]
        _out = mo.ui.tabs({"per job": mo.ui.table(job_table.round(dict.fromkeys(job_table.select_dtypes("number").columns, 1)), selection=None),
                           "per node": mo.ui.table(_by_node.sort_values("tiles_per_core_hour").round(1).reset_index(),
                                                   selection=None)})
    else:
        _out = mo.md("No jobs listing for this run (`jobs.txt` in the run folder, or `sacct` on Goethe).")
    _out
    return (job_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Totals, and what the full plan would cost

    The projection takes this run's own plan (`plan/tiles.csv` and the periods in
    `run/config.yaml`) at the measured pace. The assumptions below are editable: change the
    number of nodes or the Earth Engine cost per tile and the table follows. The measured
    Earth Engine cost for a month is on the Earth Engine configuration page in the Cloud
    console, divided by the tile-periods scored that month.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    settings = mo.ui.dictionary({
        "nodes": mo.ui.number(1, 20, value=3, label="nodes at once"),
        "cores": mo.ui.number(1, 128, value=40, label="cores per node"),
        "eecu_quarterly": mo.ui.number(1, 1000, value=145, label="EECU-seconds per tile, quarter"),
        "eecu_monthly": mo.ui.number(1, 1000, value=50, label="EECU-seconds per tile, month"),
        "quota": mo.ui.number(1_000, 1_000_000, step=1_000, value=100_000, label="Earth Engine quota, EECU-hours a month"),
        "allocation": mo.ui.number(1_000, 10_000_000, step=1_000, value=200_000, label="Goethe allocation, core-hours"),
    })
    mo.accordion({"Assumptions": settings})
    return (settings,)


@app.cell(hide_code=True)
def _(RUN, job_table, load_config, mo, pd, perf, settings, tasks):
    _s = settings.value
    _t = perf.totals(tasks, job_table, _s["cores"])
    _per_core_hour = _t.get("tile_periods_per_core_hour")
    _notes = []
    if not _per_core_hour and _t.get("task_hours"):
        _per_core_hour = _t["tile_periods_saved"] / (_t["task_hours"] * _s["cores"] / 4)   # four tasks share a node
        _notes.append(f"No jobs listing, so the pace is estimated from the tasks' own hours: "
                      f"{_per_core_hour:,.0f} tile-periods per core-hour. This leaves out each node's idle tail and is optimistic.")
    _totals = pd.DataFrame({"value": _t}).rename_axis("total").reset_index()
    _proj = None
    if _per_core_hour and (RUN / "run" / "config.yaml").is_file():
        _cfg = load_config(RUN / "run" / "config.yaml")
        _n_tiles = len(pd.read_csv(RUN / "plan" / "tiles.csv"))
        _n_periods = {f: len(p) for f, p in _cfg.periods.items()}
        _notes.append(f"This run's plan: {_n_tiles:,} tiles, " + ", ".join(f"{n} {f} periods" for f, n in _n_periods.items()) + ".")
        _proj = perf.projection(_n_tiles, _n_periods, _per_core_hour,
                                {"quarterly": _s["eecu_quarterly"], "monthly": _s["eecu_monthly"]},
                                _s["cores"], _s["nodes"], _s["quota"], _s["allocation"])
    mo.vstack([mo.ui.table(_totals, selection=None, pagination=False),
               *[mo.md(n) for n in _notes],
               mo.ui.table(_proj.round(2), selection=None, pagination=False) if _proj is not None else mo.md("")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `core_hours` is what the plan costs of the Goethe allocation at this pace, and
    `days_on_N_nodes` the calendar time with that many nodes at once (about three fit Earth
    Engine's 40 concurrent requests). `months_of_quota` is how many months of the Earth Engine
    quota the plan needs. Above one month the project runs in Earth Engine's slower restricted
    mode, so the plan either takes longer, is spread over several months, or shrinks.
    """)
    return


if __name__ == "__main__":
    app.run()
