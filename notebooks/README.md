# The notebooks

Four marimo notebooks, each a plain Python file that opens as an interactive page:

| notebook | what it shows | for |
|:--|:--|:--|
| `panel_overview.py` | the chip panel: variables, distributions, checks, detections over time, cloud cover, a map, any chip through time | everyone |
| `amw_comparison.py` | our quarters against AMW's published detections, chip by chip | everyone |
| `explore_runs.py` | the pipeline's quality control: task checks, the chip grid, border tiles, the panel against detect's output | Ruby |
| `run_performance.py` | how fast a run went, what limited it, what the full plan costs | Ruby |

They read a run folder and never change it. Coauthors see the first two; Ruby's view also
lists her two. The quarter, the detection rule and the threshold are dropdowns and a
slider at the top of each page.

## On Goethe: viewing in the browser

Step by step, for coauthors: [GUIDE.md](GUIDE.md). In short: `sbatch cluster/view.sbatch
[<run>]` starts the notebooks on a compute node for two hours; `~/marimo-view-<jobid>.out`
names the node and port to forward (VS Code's Ports tab, or the `ssh -N -L` line it prints)
and the address to open in a normal browser.

## On a laptop, or to edit a notebook

```bash
uv sync --locked                                              # once: the environment, from uv.lock
uv run marimo edit notebooks/panel_overview.py -- --run tests_output/<run>    # edit
uv run marimo run notebooks/ -- --run tests_output/<run>                       # view all
```

`python panel_pipeline/checks/tests/fake_run.py tests_output/fake_run` makes a small
made-up run to try them on.

## How they stay fast

Sections that read every period of the run (the checks, cloud cover per period, months
against quarters, the distributions) are computed by `cluster/summaries.sbatch`, which
runs the notebooks as scripts, and stored in `<run>/summaries/`. In the browser they are
only read; before they are stored they say *Not stored yet*, so a page never sits
computing for many minutes. A stored summary is used until the run changes; run
`summaries.sbatch` after each new period. (`explore_runs` still computes for its owner
in the browser, and in `marimo edit` everything is computed.)

Charts with one point per chip draw at most 10,000 points (a random sample, said in the
chart's title); a browser cannot draw a full run's chips, and marimo refuses outputs over
8 MB. The counts in the tables are always over every chip.
