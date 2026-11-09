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

You need a Goethe login and a terminal on your laptop (on a Mac: the Terminal app). No copy
of the code, no Python, no git: everything runs on Goethe with Ruby's shared environment and
Ruby's copy of the code, and the data never leaves the cluster.

1. **Start the notebooks.** Log in (`ssh <your user>@goethe.hhlr-gu.de`, or a VS Code
   window connected to Goethe) and run:
   ```bash
   sbatch /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch
   ```
   This asks the cluster for a compute node for two hours. Wait half a minute, then:
   ```bash
   cat $(ls -t ~/marimo-view-*.out | head -1)
   ```
   It prints one `ssh -N -L ...` line and an address.
2. **Connect your laptop to it.** Open a *second* terminal on your laptop, paste the
   `ssh -N -L ...` line, type your password. Nothing more is printed; that is right, leave
   the window open. If it says *Address already in use*, change the first number in the
   line to any other between 20000 and 60000, and use that number in the address too.
3. **Open the address** (`http://localhost:...?access_token=...`) in your normal browser,
   not VS Code's preview. You see the list of notebooks; click one.

The first time a notebook opens for a run it can take a minute or two, while it reads the
run (a spinner shows what it is doing). When you are done, close the browser tab and the
`ssh` window; the job ends by itself after two hours, or sooner with `scancel <job number>`.
For a longer session:
`sbatch --partition=general1 --time=06:00:00 /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch`.
A different run: add its folder at the end of the `sbatch` line.

Every table has an *Export* button (CSV), and every chart a menu to save it as an image.

## On a laptop, or to edit a notebook

```bash
uv sync --locked                                              # once: the environment, from uv.lock
uv run marimo edit notebooks/panel_overview.py -- --run tests_output/<run>    # edit
uv run marimo run notebooks/ -- --run tests_output/<run>                       # view all
```

`python panel_pipeline/checks/tests/fake_run.py tests_output/fake_run` makes a small
made-up run to try them on.

## How they stay fast

Sections that read the whole run (the checks, cloud cover per period, months against
quarters, the task statistics) store their result in `<run>/summaries/` the first time
the run's owner opens them, or when `cluster/summaries.sbatch` runs. A stored summary is
used until the run changes. A viewer without write access who opens a section before its
summary exists gets it computed in memory instead; that works, but takes a while on a
full run and is not stored, so run `summaries.sbatch` after each new period.

Charts with one point per chip draw at most 10,000 points (a random sample, said in the
chart's title); a browser cannot draw a full run's chips, and marimo refuses outputs over
8 MB. The counts in the tables are always over every chip.
