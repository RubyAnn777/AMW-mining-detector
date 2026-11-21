# The notebooks

Marimo notebooks, each a plain Python file that opens as an interactive page. They read a
run folder and never change it, except that `build_tables.py` writes `<run>/analysis/`.

| notebook | question | reads |
|:--|:--|:--|
| `1_clouds.py` | How cloudy was each period and place, and do detections depend on it? Which cloud measure separates missed from detected periods of the same mine, and where would a "cloudy" cut-off sit? | `analysis/` |
| `2_persistence.py` | Do mines stay detected from one period to the next? When they drop out (D M D, D M M D, ...), was the missed period cloudier? How does AMW's yearly two-year rule treat mines that start late in a year? | `analysis/` |
| `3_distance.py` | How spread out are the scores (across chips, across the six models, over time)? Are non-detections near misses or clear zeros? How do score and detection change with the distance to the nearest known mine? | `analysis/` |
| `4_frequencies.py` | What do months, quarters and AMW's quarters and years each detect, and how do they add up? Chip by chip: what did a month miss that its quarter found, and why? | `analysis/` |
| `panel_overview.py` | The chip panel itself: periods, an example of every variable, the checks, counts per rule, a map, chips through time | the panel |
| `amw_comparison.py` | Our quarters against AMW's published quarters, chip by chip | `amw_comparison/` |
| `amw_history.py` | Would scoring AMW's known mining tiles first have caught our detections? | `amw_comparison/` |

Every notebook lists the variables it uses at the top (open "The variables in this
notebook"), defined once in `panel_pipeline/lib/variables.py`, and every figure has a
numbered caption saying what one mark is, what each axis and colour shows, and which chips
and periods it covers. Subregions, detection rule and frequency are chosen at the top.

## Folders

```
notebooks/
  1_clouds.py ... 4_frequencies.py    the analysis notebooks
  panel_overview.py, amw_comparison.py, amw_history.py
  build_tables.py                     builds <run>/analysis/ for notebooks 1 to 4
  analysis.yaml                       settings: which years count as known mines
  archive/                            retired notebooks (explore_runs, run_performance), kept to look back
<run>/analysis/                       written by build_tables.py; small (tens of MB)
  static/                             chips, tiles, AMW's files, known mines, followed chips
  quarterly/period=<tag>/, monthly/period=<tag>/   per-period summaries
reports/                              static HTML exports to share (not committed)
```

## The procedure

### 1. Build the analysis tables

Notebooks 1 to 4 do not read the 145 million chips of a period each time they open. A
builder reads every period once and keeps what the notebooks need: per tile and per
0.25° cell, histograms of every score and cloud measure, counts per score, cloud and
distance band, a fixed sample of one chip in 1,000, and the full rows of every chip that
ever scores 0.2 or more, is in one of AMW's files, or is a known mine. Counts of
detections are therefore exact, never sampled.

```bash
uv run python notebooks/build_tables.py --run <run folder>
```

AMW's raw files (`amazon_basin_<2018..2025|Q125..>_t0.4.geojson`) are read from
`amw_comparison/data/amw/` (another folder: `--amw-dir`). The first build takes about
2 to 3 minutes per period on a full run (about 40 minutes for 7 quarters and 9 months).
After that only new or reassembled periods are read.
`--rebuild` starts over.

**Where.** On Goethe, `cluster/summaries.sbatch <run>` runs the builder before storing the
other notebooks' summaries. **While Goethe is down (23 Nov to 20 Dec 2026)**, run it on the
Mac against the copy of the run:

```bash
cd ~/RA_Raji/AMW-mining-detector
uv sync --locked
uv run python notebooks/build_tables.py --run cluster_output/aca_full_2026-11-01
```

### 2. Copy (only if the tables were built elsewhere)

`<run>/analysis/` is all notebooks 1 to 4 need, so copying it is enough:

```bash
rsync -av goethe:/work/goldmine/grambauer/runs/<run>/analysis/ cluster_output/<run>/analysis/
```

### 3. Open

```bash
uv run marimo run notebooks/1_clouds.py -- --run cluster_output/<run>     # view
uv run marimo edit notebooks/1_clouds.py -- --run cluster_output/<run>    # view and edit
uv run marimo run notebooks/ -- --run cluster_output/<run>                 # all, with a list
```

On Goethe, `sbatch cluster/view.sbatch [<run>]` serves them in the browser
([GUIDE.md](GUIDE.md)).

### 4. Export static versions to share

```bash
mkdir -p reports
for NB in 1_clouds 2_persistence 3_distance 4_frequencies; do
  uv run marimo export html --no-include-code notebooks/$NB.py -o reports/$NB.html -- --run cluster_output/<run>
done
```

A static file opens in any browser without Python. What still works in it: hovering, the
maps' period slider and play button, zooming charts. What is frozen: marimo's own
controls (dropdowns, sliders, the box selection in notebook 4), which keep the values
they had at export. To share another subregion or rule, change the default at the top of
the notebook, or export from `marimo edit` after choosing them. The map backgrounds load
from the internet when the file is opened; the data are inside the file.

### When a new period is assembled

Run step 1 again (only the new period is read), then export. Every list of periods,
slider and comparison in the notebooks follows the periods present, so nothing in the
notebooks needs changing; a quarter whose three months exist is automatically compared
with them. When all four quarters of a year are assembled, move the year from `amw_years`
to `our_years` in `analysis.yaml` to measure distances from our own detections instead of
AMW's for that year.

## Trying them without a run

```bash
python panel_pipeline/checks/tests/fake_panel.py tests_output/fake_panel     # a made-up panel and AMW files
python notebooks/build_tables.py --run tests_output/fake_panel --amw-dir tests_output/fake_panel/amw
uv run marimo edit notebooks/1_clouds.py -- --run tests_output/fake_panel
```

`panel_pipeline/checks/tests/fake_run.py` makes the small run `panel_overview` and
`amw_comparison` are tested on. Tests: `python -m unittest discover -s panel_pipeline/checks/tests`.

## How panel_overview and amw_comparison stay fast

Sections that read every period (the checks, the distributions) are computed by
`cluster/summaries.sbatch`, which runs those notebooks as scripts, and stored in
`<run>/summaries/`. In the browser they are only read; before they are stored they say
*Not stored yet*. Charts with one point per chip draw at most 10,000 points (a random
sample, said in the chart's title); the counts in the tables are over every chip.
