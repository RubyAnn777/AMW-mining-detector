# The notebooks

Four marimo notebooks, each a plain Python file that opens as an interactive page:

| notebook | what it shows | for |
|:--|:--|:--|
| `panel_overview.py` | the chip panel: variables, distributions, checks, detections over time, cloud cover, a map, any chip through time | everyone |
| `amw_comparison.py` | our quarters against AMW's published detections, chip by chip | everyone |
| `explore_runs.py` | the pipeline's quality control: task checks, the chip grid, border tiles, the panel against detect's output | Ruby |
| `run_performance.py` | how fast a run went, what limited it, what the full plan costs | Ruby |

They read a run folder and never change it. Settings (which run, which quarter, which
detection rule, ...) are dropdowns and sliders at the top of each page.

## On Goethe: viewing in the browser

You need a Goethe login, VS Code with the **Remote - SSH** extension, and nothing else: no
copy of the code, no Python setup, no git. Everything runs on Goethe with Ruby's shared
environment and Ruby's copy of the code, and the data never leaves the cluster.

1. **Connect.** In VS Code, press F1 (or Cmd/Ctrl-Shift-P), choose *Remote-SSH: Connect to
   Host...*, type `<your user>@goethe.hhlr-gu.de`, enter your password. A new window opens,
   connected to Goethe.
2. **Start the notebooks.** In that window, *Terminal → New Terminal*, then:
   ```bash
   sbatch /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch
   ```
   This asks the cluster for a compute node for two hours. After about 20 seconds:
   ```bash
   cat ~/marimo-view-*.out
   ```
   shows the node, a port, and an address.
3. **Forward the port.** Open the *Ports* tab (next to *Terminal*), click *Forward a Port*,
   and type the `node:port` from the file, for example `node17-003:21234`.
4. **Open the address** from the file (`http://localhost:...?access_token=...`) in your
   browser. You see the list of notebooks; click one.

When you are done, close the browser tab; the job ends by itself after two hours, or
sooner with `scancel <job number>`. For a longer session:
`sbatch --partition=general1 --time=06:00:00 /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch`.

A different run: add its folder at the end of the `sbatch` line, or pick it in the *Run*
dropdown at the top of a notebook.

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
summary exists gets a button to compute it in the browser instead; that works, but takes
long on a full run and is not stored.
