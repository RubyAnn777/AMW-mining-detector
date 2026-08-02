# The Time-Series Code, Explained — A Lecture Course

*Written for a reader who has never worked with shells, Git, or scientific
Python. Companion documents: `CODE_WALKTHROUGH.md` (the upstream pipeline),
`pipeline_spatial_card.md` (the model), `GIT_PLAYBOOK.md` (Git recipes).
This course explains the code **we** wrote in `timeseries/`, the command-line
machinery around it, the GitHub "hub" your fork lives in, and what was changed
in the July 27 working session.*

**Syllabus**

- Lecture 0 — The machine room: terminal, shell, CLI anatomy, environments
- Lecture 1 — The hub: how Git and GitHub actually work, and what we did to your fork
- Lecture 2 — The scientific problem the new code solves
- Lecture 3 — `coverage.py`: the cloud flag (and Earth Engine's deferred computation)
- Lecture 4 — `panel.py`: the chip panel (threads, alignment, defensive programming)
- Lecture 5 — `run_trials.py`: orchestration and a deep dive on how a CLI is built
- Lecture 6 — `run_overnight.sh`: shell scripting, processes, and surviving the night
- Lecture 7 — `analysis.py`: the statistics, each measure justified and criticized
- Lecture 8 — The session changelog: everything that was done and why

---

## Lecture 0 — The machine room

### 0.1 Terminal, shell, and why typed commands at all

The **terminal** is a window that sends your keystrokes to a program called a
**shell** (on your Mac: `zsh`; our scripts use its close relative `bash`). The
shell reads a line, splits it on spaces, treats the first word as a program to
run and the rest as **arguments** handed to that program. That's the entire
trick. When you type:

```bash
python timeseries/run_trials.py --months 2025-07 --limit_tiles 2
```

the shell finds the program `python`, starts it, and passes it the list
`["timeseries/run_trials.py", "--months", "2025-07", "--limit_tiles", "2"]`.
The shell does not know what `--months` means — only the Python program does
(Lecture 5). Graphical apps hide this; science prefers it exposed, because a
typed command can be *saved, repeated, and put in a paper's appendix*. A click
cannot. Reproducibility is the whole reason this course exists.

**Anatomy of a command line** (the grammar you'll see everywhere):

```
program  subcommand  --option value  --flag  positional_argument
git      fetch       (none)          (none)  upstream
python   script.py   --year 2025     --force (none)
```

Options starting with `--` are named and order-independent; a **flag** is an
option that takes no value (its presence means "yes"); **positional** arguments
are identified by their position alone.

### 0.2 Three streams and exit codes

Every running program is wired to three channels: **stdin** (input), **stdout**
(normal output), **stderr** (errors/progress). The `>` symbol redirects stdout
to a file; `2>&1` says "send stderr to the same place as stdout" — you'll see
exactly this in `run_overnight.sh`, because overnight we want *everything* in
one log file. When a program ends it returns an **exit code**: 0 means success,
anything else means failure. The shell operator `&&` means "run the next
command only if the previous one returned 0" — that is how the overnight script
chains *trials then analysis* without running the analysis on broken data.

### 0.3 Environment variables

An **environment variable** is a named string a process inherits from whoever
started it — a way to pass configuration without editing code. Our pipeline
needs `EE_PROJECT` (your Google Cloud project name) because `core/gee.py` reads
it at import time: `os.environ.get("EE_PROJECT", "earthindex")`. Your `.env`
file stores it on disk; the launcher script copies it into the environment
(`export EE_PROJECT=...`) before starting Python. The cute error you hit
(`EE_PROJECT: command not found`) was the shell trying to *execute* the line
`EE_PROJECT = value` — with spaces around `=`, the shell reads `EE_PROJECT` as
a program name. Shell assignment permits no spaces. The fixed launcher now
extracts the value with a text tool (`sed`) instead of executing the file.

### 0.4 Virtual environments

Python packages are installed system-wide by default, which turns every project
into a fight over versions. A **virtual environment** (`venv/` in the repo) is
a private, per-project copy of the Python interpreter plus its packages.
`source venv/bin/activate` simply puts the venv's `bin/` folder first on your
`PATH` (the list of folders the shell searches for programs), so that `python`
now means *this project's* Python, with TensorFlow, geopandas, earthengine-api
etc. at the exact pinned versions in `requirements-lock_2026-07-17.txt`. Forget
to activate and you get `ModuleNotFoundError` — the single most common error in
this line of work.

### 0.5 How Python runs a file

`python file.py` executes the file top to bottom, once. `def` and `class`
statements don't *run* the code inside them — they *define* objects for later
use. `import gee` executes `gee.py` once, caches the resulting module, and
binds the name `gee`. Where does Python look for `gee.py`? In the folders
listed in `sys.path`. Our scripts *prepend* the repo's `core/` folder to that
list (`sys.path.insert(0, ...)`) so the upstream modules are importable no
matter which folder you launched from — a small trick that removes a whole
class of "works in one directory, breaks in another" bugs.

---

## Lecture 1 — The hub: Git, GitHub, and your fork

### 1.1 What Git actually stores

Git is a **content-addressed history database** that lives inside the hidden
`.git/` folder of the repo. A **commit** is a frozen snapshot of every tracked
file, plus a pointer to its parent commit — so history is a chain (in general a
graph, since two chains can merge). Each commit is named by a hash like
`36bec54`. A **branch** is nothing more than a movable label pointing at one
commit; `main` is just the conventional name for the principal label. Nothing
is ever edited in place: committing adds a new snapshot and moves the label.
That is why experimentation is safe — the old snapshots remain reachable
(your `backup-before-upstream-2026-07-24` branch is exactly such a parked label).

### 1.2 GitHub: the hub in "hub"

Git itself is fully local. **GitHub** is a hosting service where repositories
can be published so multiple people (or multiple machines) can exchange
commits. Your local repo knows two published repos, called **remotes**:

```
origin    https://github.com/RubyAnn777/AMW-mining-detector      (your fork)
upstream  https://github.com/earthrise-media/mining-detector     (theirs)
```

A **fork** is GitHub's one-click copy of someone else's repository into your
account. Crucially, *it is a copy from that moment* — it never updates itself.
The Amazon Mining Watch team kept committing after you forked; those commits
sat in *their* repo until you explicitly downloaded them with `git fetch
upstream`. This is the answer to "does a fork auto-update?": no, and `git pull`
didn't help because `pull` only consults the remote your branch tracks —
`origin`, your own fork, which indeed had nothing new. Hence "Already up to
date" while upstream was 400 commits ahead.

### 1.3 Fetch, merge, and why we refuse to merge

`git fetch upstream` downloads their commits and updates your local *copies* of
their branch labels (`upstream/main` etc.) — it touches none of your files. The
textbook next step would be `git merge upstream/main`, which replays their
changes onto yours. We deliberately do **not** do this. You archived hundreds
of files (old models, website code); a merge reconciles *entire histories*, so
every archived file that upstream later touched would be resurrected into your
working tree. Instead we use a surgical alternative:

```bash
git checkout upstream/main -- core/
```

meaning "copy the folder `core/` exactly as it exists in their `main` into my
working tree" — files you never name can never come back. This is the entire
sync strategy, formalized in `GIT_PLAYBOOK.md`.

### 1.4 Visibility and safety

Commits you never push exist only on your disk. Commits pushed to `origin` are
on your public fork — browsable in principle, invisible in practice, and in no
way inserted into upstream's repository, history, or notifications. The only
mechanism that puts your work in front of the owners is a **Pull Request**,
which is always an explicit act. As a hard guarantee, this repo's upstream
*push URL* is set to the literal string `DISABLE`, so even a mistyped `git push
upstream` fails — and because remotes are repo-level, that covers every branch,
including `main` after their PR #29 merge. Their MIT license explicitly permits
all of this modifying-and-keeping-a-fork activity.

### 1.5 What we did to the hub state (summary; full list in Lecture 8)

Audited your fork against `upstream/main` after their July 24 merge: `core/`
was already byte-identical except a 13-line stale `core/README.md` (synced);
your model weights + config matched; `scripts/` and `utils/` had no upstream
changes since your copies. Discovered the legacy `gee/` folder — deleted
upstream, still live in your tree with *older* code — and archived it to
`_archive/gee_pre-core_2026-07-27/`, removing it from Git tracking. Recovered
one useful missing file (the `t0.43` config for your exact ensemble). Nothing
you archived was resurrected; nothing was pushed anywhere.

---

## Lecture 2 — The scientific problem the new code solves

### 2.1 One-paragraph recap of the detector

For any region and date window, the pipeline lays ~5.5 km **tiles** over the
area, asks Google Earth Engine (GEE) for a cloud-masked **median composite** of
all Sentinel-2 satellite scenes in the window (13 spectral bands, 10 m pixels,
returned as a 576×576×13 number grid per tile), slides a 48-pixel (=480 m)
window across each tile in 24-pixel steps, and lets a convolutional neural
network score every such **chip** with a probability that it contains mining.
Details and citations: `pipeline_spatial_card.md`.

### 2.2 The silent failure we had to fix first

Cloud masking happens per scene; the median then averages over the surviving
clear pixels. If a pixel was cloudy in *every* scene of the window, nothing
survives — and Earth Engine transmits a **0** in all 13 bands. An all-zero chip
fed to the CNN scores ≈ 0: "no mine". So the raw output *cannot distinguish "we
looked and saw nothing" from "we never saw the ground"*. Over the Amazon in the
rainy season this is not a corner case; a monthly window (~6 satellite passes)
can easily be fully clouded. Any time series built naively on this output would
show mining "disappearing" every wet season. This is a textbook **missing-data
problem**, and the first deliverable — the **cloud flag** — makes the
missingness *observable* so the later econometrics can model it rather than
absorb it as fake signal.

### 2.3 The experiment the code runs

Same area (La Pampa corridor, ~1,900 km² of dense artisanal mining), same
model, same year (2025), two measurement cadences: 12 monthly composites and 4
quarterly composites. For every chip and every window we record the CNN
probability *and* the cloud flag. Lecture 7's analysis then asks: how much is
missing, when, and where; does apparent change track clouds; do the two
cadences agree; where does monthly noise live. The design principle throughout:
**record continuously, decide late** — store all probabilities, threshold only
at analysis time.

---

## Lecture 3 — `coverage.py`: the cloud flag

Open the file alongside this text. We go block by block; every line belongs to
one of the blocks below.

### 3.1 The preamble (lines 1–40): docstrings as lab notebooks

The triple-quoted text at the top is a **docstring** — a string Python attaches
to the module as documentation. Ours is deliberately long: it states *why the
module exists* (the silent zero-fill problem), *what it computes* (N_CLEAR,
N_TOTAL), and *what would falsify its assumptions* (the fill=0 check). In
academic code the docstring is your methods section; six months from now it is
the difference between reusing the module and rewriting it.

`from __future__ import annotations` makes type hints lazy — a compatibility
nicety that lets us write modern hints like `str | None` on older Pythons.
**Type hints** (`start_date: str`, `-> np.ndarray`) are annotations only:
Python does not enforce them, but they document intent and let editors catch
mistakes. `logging.getLogger(__name__)` creates a named channel for progress
messages; unlike `print`, logging can be globally routed to files with
timestamps — which the overnight runner does.

The imports tell the dependency story: `ee` is Google's Earth Engine client;
`rasterio` reads/writes GeoTIFFs; and — important design decision — we import
`_ensure_earth_engine_initialized` and `ensure_tile_shape` **from the upstream
modules** rather than copying their code. One authoritative implementation,
zero drift. The trade-off is that upstream renames break us, which is why
`panel.py` pins and asserts (Lecture 4.4).

### 3.2 `class CoverageExtractor` and `__init__`

A **class** bundles data (attributes) with functions that use them (methods).
We mirror upstream's `GEE_Data_Extractor` shape deliberately: construct with a
date window, hold a prepared server-side image, offer a per-tile fetch method.
`__init__` runs at construction. Note `str(start_date)` — cheap defensive
coercion so a `datetime.date` passed by accident won't corrupt cache filenames.
`self.cache_dir.mkdir(parents=True, exist_ok=True)` creates the cache folder
and, thanks to `exist_ok`, is **idempotent** — safe to run twice. Idempotence
is a theme in this codebase: overnight jobs die and restart, so every operation
should be repeatable without harm.

### 3.3 `_build_count_image`: deferred computation, the key GEE concept

This method contains *no data*. Every `ee.*` call builds a node in a
**computation graph** that lives on Google's servers; nothing executes until a
result is explicitly requested (in `get_tile_counts`). This "describe now,
compute later" pattern — *lazy* or *deferred* evaluation — is how GEE lets a
laptop command a planetary archive: your machine sends a small recipe, Google's
cluster does the work.

The recipe itself mirrors upstream's composite line by line, and must — the
counts have to describe *the same composite the CNN saw*:

- `ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(...)` — all
  Sentinel-2 scenes touching the window (end-exclusive!).
- `.linkCollection(cs_plus, ["cs_cdf"])` — attach Google's Cloud Score+ quality
  band to each scene.
- `.map(lambda img: img.updateMask(...gte(threshold)))` — `map` applies a
  function to every scene; a **lambda** is an inline unnamed function. This
  *masks* (invalidates) every pixel scoring below 0.6 clear-sky probability.
- `masked.select("B2").count()` — the crux. `count()` collapses the scene stack
  to one image whose per-pixel value is *how many scenes were valid there*. On
  the masked stack that is **N_CLEAR** (scenes surviving the cloud mask =
  sample size of the median); on the raw stack it is **N_TOTAL** (scenes whose
  footprint covered the pixel). Their ratio is the local clear-sky rate.
- `.unmask(0).toUint16()` — where zero scenes existed the count is itself
  masked; `unmask(0)` makes the zero explicit rather than trusting transport
  defaults, and `toUint16` picks a compact integer type (0–65535) since counts
  are small non-negative integers.

### 3.4 The cache trio: `_cache_path`, `_save`, `_load`

Network fetches are the expensive resource (seconds each, thousands per night),
so results are **memoized on disk**: the filename
`S2COV_{tile.key}_{start}_{end}.tif` encodes the *complete identity* of the
computation — same tile, same window ⇒ same file, so a re-run is a free disk
read. Cache-key design is the whole game: forget the date in the name and a
January raster would silently answer a July query.

`_save` writes a **GeoTIFF** — an ordinary TIFF image carrying georeferencing
metadata: a **CRS** (which coordinate system: EPSG:4326, plain lat/lon) and an
**affine transform** (six numbers mapping pixel row/col → geographic x/y;
`from_bounds` derives them from the tile's corner coordinates and size). With
those two, QGIS can drop the raster onto a map in the right place. Note the
axis gymnastics: our arrays are `(H, W, bands)` (image convention), rasterio
wants `(bands, H, W)` — `np.moveaxis` converts on the way out, `_load` converts
back on the way in. Getting this wrong produces beautifully scrambled rasters.

### 3.5 `get_tile_counts`: the only method that touches the network

Read it as: *check cache → build request → execute → normalize shape → write
cache → return.* The execution line, `ee.data.computePixels({...})`, is the
moment the deferred graph actually runs; `clipToBoundsAndScale(geometry, width,
height)` forces the answer onto the exact 576×576 grid of the imagery tile, so
pixel *(i, j)* in the coverage raster is pixel *(i, j)* in the image — the
alignment everything in Lecture 4 relies on. GEE returns a **structured
array** (a table-like array with named fields, one per band); the
`raw.dtype.fields` branch detects that and `np.stack` converts it to a plain
`(H, W, 2)` float array. `ensure_tile_shape` (upstream's helper) pads any
short-by-one-pixel responses. The `try/except` around cache writes embodies a
principle: *a failed cache write must never fail the science* — log the
warning, return the data.

### 3.6 `chip_coverage_stats`: from pixels to a per-chip flag

`cov[row:row+chip, col:col+chip]` is **slicing** — a zero-copy view of one
48×48 window. The elegant line is `np.mean(n_clear > 0)`: the comparison
produces a Boolean array; Python treats `True` as 1, so the mean of a Boolean
array *is* the fraction of pixels satisfying the condition. That single number
is `clear_frac`, the chip's cloud flag: 1.0 = fully observed, 0.0 = the chip is
pure fill. Alongside it we return the mean and min of N_CLEAR (precision of
the median — Lecture 7.H uses this) and mean N_TOTAL.

---

## Lecture 4 — `panel.py`: the chip panel

### 4.1 Why a "panel" (the measurement-theory argument)

Upstream's pipeline writes only chips whose score *exceeds a threshold* —
sensible for map-making, wrong for statistics, for three reasons. First,
thresholding throws away the continuous score, and Lecture 7 shows the noise
lives near the threshold; you want to threshold *late*, on smoothed series.
Second, absence becomes ambiguous: a chip missing from the file might be
negative, or clouded, or failed. Third, its cumulative-append output format
mixes runs. A **panel** — econometrics jargon for observations of the *same
units* across *multiple periods* — fixes all three: every chip appears in
every period with its probability and its cloud flag. Balanced by
construction: tiling depends only on the AOI, so the ~37,000 chip IDs are
identical in all 16 files.

### 4.2 `chip_windows`: 21 lines about a subtle bug that never happened

The upstream chipper `cut_chips` returns chips and their geo-rectangles but
*not* their pixel offsets, which we need to slice the coverage raster. So we
re-derive the offsets — and here hides the subtlest trap in the whole build:
`zip(windows, probs)` pairs the i-th window with the i-th probability *by
position*. If our loop visited windows in any other order than `cut_chips`
does, every chip would be silently assigned some *other* chip's cloud flag —
no crash, plausible numbers, corrupted science. Defense in depth: (1) the loop
is a verbatim copy of upstream's (outer loop over columns, inner over rows —
note `range(0, W - chip + stride, stride)` yields 23 positions per axis, hence
23² = 529 chips per tile); (2) a runtime `assert` checks the counts match; (3)
during the session we *proved* order-identity by executing both loops side by
side. A **generator** (`yield`) produces pairs one at a time without building
a list — a convenience, not a necessity.

### 4.3 `fill_zero_mismatch`: falsifiability in eleven lines

The pipeline card assumed (flagged `[UNVERIFIED]`) that fully-masked pixels
arrive as 0. This function *tests* that claim on every tile, every night:
`valid_from_counts` (N_CLEAR > 0, from the independent coverage request) versus
`valid_from_image` (`~np.all(pixels == 0, axis=-1)`, i.e. not-all-13-bands-zero
— `~` is Boolean NOT, `axis=-1` collapses the band axis). `np.mean(a != b)` is
the disagreement rate; `!=` on Booleans is exclusive-or. If the assumption
holds it is ≈ 0 forever; if Earth Engine ever changes its fill behaviour, the
number jumps and Report §B catches it. Turning a footnote assumption into a
monitored quantity is cheap insurance.

### 4.4 `PeriodSpec` and the engine setup

`@dataclass` auto-generates boilerplate (constructor, printable repr) for a
class that is pure data — `PeriodSpec` is just (label, start, end, cadence)
with the **end-exclusive** convention documented at the definition site.

`build_period_panel` then assembles upstream machinery: read the AOI GeoJSON
(the `union_all`/`unary_union` try/except tolerates two geopandas versions —
graceful degradation), tile it with *identical parameters to production*
(552 px + 12 px pad = 576), and construct `InferenceEngine` with
`embedding_strategy="none"` — the flag that selects the from-scratch CNN path
and no foundation model. Then three `assert hasattr(engine, ...)` lines: we
call private-by-convention methods (Python's `_underscore` prefix is a social
contract, not a lock), which upstream may rename. The asserts convert that
future risk from *silently wrong* to *loudly broken at startup*, with a message
saying exactly what to re-pin. This is **defensive programming**: make the
failure mode a crash, never a plausible-looking number.

### 4.5 The batch loop: threads, futures, and the GIL

Per batch of 8 tiles we fetch imagery and coverage **concurrently** with
`ThreadPoolExecutor`. Background: Python's **GIL** (Global Interpreter Lock)
prevents two threads from executing Python bytecode simultaneously — so
threads do *not* speed up computation. But a thread *waiting on the network*
releases the GIL; for I/O-bound work (our fetches are seconds of waiting per
tile), 8 threads ≈ 8× throughput. `ex.submit(fn, arg)` schedules a call and
returns a **future** — a receipt; `.result()` waits for and returns the value.
The dict comprehensions keyed by `tile.key` keep results identified. Inference
stays *outside* the pool: TensorFlow manages its own parallelism, and one tile
in RAM at a time bounds memory (~17 MB per tile tensor).

The fetch wrappers `_fetch_pixels_safe` / `_fetch_cov_safe` never raise — they
return `None` and log. Inside a thread pool an uncaught exception surfaces only
when `.result()` is called, in confusing places; converting failures to `None`
at the source makes the per-tile skip logic explicit: failed tiles are recorded
as `status=FAILED` in the tile stats, and the analysis treats them as
*missing*, never as zeros. The same taxonomy as clouds: absence of evidence,
recorded as absence.

### 4.6 Per-tile science and the write-out

For each fetched tile: `np.clip(pixels / 10000, 0, 1)` — Sentinel-2 stores
reflectance ×10,000, the CNN was trained on 0–1, and this line is verbatim from
upstream's `predict_on_tile_pixels`, which matters: preprocessing must match
training bit-for-bit. `cut_chips` slices 529 chips; `engine._model_infer`
(thread-locked TensorFlow predict) scores all of them in one batched call;
`_probs_from_predictions` averages the ensemble members into one probability
per chip. Then the zip pairs each probability with its window's coverage stats
into one row dict. `chip_id = f"{tile.key}:c{i}"` is stable across periods —
the join key of the entire downstream analysis.

Rows become a `GeoDataFrame` (a pandas table with a geometry column + CRS)
written as **GeoParquet** — a compressed columnar format that loads ~100×
faster than GeoJSON and preserves types exactly; one file per period is the
resume-safety unit. Side outputs per period: a tile-stats CSV (timings for
cost extrapolation, coverage, the fill-check) and a thresholded GeoJSON purely
for dragging into QGIS — the parquet, not it, is the dataset.


---

## Lecture 5 — `run_trials.py`: orchestration, and how a CLI is built

### 5.1 The orchestrator pattern

Science code splits naturally into *mechanism* (panel.py: how to measure one
period) and *orchestration* (run_trials.py: which periods, in what order, what
happens on failure). Keeping them in separate files means the mechanism is
testable alone and the orchestration is readable as a plan: build the period
list, loop, skip what's done, contain failures, report.

### 5.2 The header: shebang and path bootstrap

Line 1, `#!/usr/bin/env python3`, is the **shebang**: if the file is executed
directly (`./run_trials.py`), the OS reads this line to learn which interpreter
to use; `/usr/bin/env python3` means "whichever python3 is first on PATH" —
i.e. the venv's, if activated. `REPO_ROOT = Path(__file__).resolve().parent.parent`
computes the repo root *from the script's own location* (`__file__` is the
script's path; `.resolve()` makes it absolute; two `.parent`s climb from
`timeseries/run_trials.py` to the root). Then the two `sys.path.insert` lines
make `core/` and `timeseries/` importable. Result: the script works from any
current directory — orchestrators must, because cron jobs and nohup sessions
start in unpredictable places.

### 5.3 Building the calendar

`month_specs` and `quarter_specs` are pure functions returning lists of
`PeriodSpec`. The **f-string** `f"{year}-{m:02d}-01"` formats with zero-padding
(`:02d` → `07` not `7`) — cache filenames and sort order depend on it. The
December/Q4 branch handles the year rollover (`2026-01-01` as the exclusive
end). Writing these five lines yourself, rather than importing a calendar
library, keeps the end-exclusive convention — the pipeline's single most
error-prone convention — visible and testable in one place.

### 5.4 argparse: the CLI deep dive

`argparse` is Python's standard machine for turning `sys.argv` (the raw word
list from the shell, Lecture 0.1) into a typed, validated object.

```python
ap = argparse.ArgumentParser(description=__doc__, ...)
ap.add_argument("--year", type=int, default=2025)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
```

Each `add_argument` declares one option: its name (`--year` becomes
`args.year`), a `type` converter (the shell delivers only strings; `type=int`
parses and *validates* — `--year banana` dies with a clean message), a
`default` used when the option is absent, and a `help` string.
`action="store_true"` declares a flag (Lecture 0.1): absent → False, present →
True. Passing `description=__doc__` recycles the module docstring as the help
text, so `python run_trials.py --help` prints the same documentation you read
in the file — one source of truth. This is also why every capability
(`--limit_tiles` for smoke tests, `--months` for subsets, `--no-quarterly`,
`--force`) is a CLI option rather than a code edit: you experiment by typing,
not by modifying the script — and your shell history becomes a lab notebook.

`--months` shows a common pattern: accept a comma-separated string, split it,
and filter the period list. Note it must match both `2025-07`-style fragments
and full labels — hence the two-clause condition with `label.lstrip("m")`.

### 5.5 Logging to two places at once

`logging.basicConfig(handlers=[StreamHandler(), FileHandler(...)])` routes every
log record to the terminal *and* a timestamped `runlog_*.log`. The format
string prepends time and severity, which is how, tomorrow, you will
reconstruct which period was slow and when the network hiccuped. Libraries
(`gee`, our modules) log into the same root logger — one night, one file.

### 5.6 Resume-safety, fault containment, exit codes

The heart of the loop is three ideas. *Resume-safety*: if the period's parquet
exists and `--force` is absent, skip — so re-running after a 3 a.m. crash costs
nothing (the file is only written at the very end of a successful period build,
so a half-finished period never masquerades as done). *Fault containment*:
`try/except` around each period converts "one bad month kills the night" into
"one bad month is logged and the other fifteen finish"; `log.exception` records
the full **traceback** (the call-stack listing showing where the error arose).
*Honest exit codes*: `return 1` if anything failed, `0` otherwise, delivered
via `raise SystemExit(main())` under the `if __name__ == "__main__":` guard —
that guard makes the file both an importable module (for tests) and a runnable
script, and the exit code is what lets `&&` in the overnight script decide
whether to run the analysis.

---

## Lecture 6 — `run_overnight.sh`: shell scripting and the night shift

### 6.1 Why a shell script at all

The overnight run is not one program but a *plan*: activate environment, load
secrets, start the long job detached, keep the machine awake, tell the human
how to watch. Plans that coordinate programs are what shell scripts are for.

### 6.2 The safety preamble

`set -euo pipefail` is the standard bash safety incantation, three switches:
`-e` exit on any failing command (default bash marches on after errors!);
`-u` treat *undefined variables* as errors (catches typos like `$LGO`);
`-o pipefail` make a pipeline fail if *any* stage fails, not just the last.
Then `cd "$(dirname "$0")/.."`: `$0` is the script's own path, `dirname` its
folder, `$(...)` is **command substitution** (run the inner command, paste its
output), so this means "cd to the repo root, wherever I was called from" — the
shell twin of Lecture 5.2.

### 6.3 Secrets without sourcing

The first version did `source .env` — *execute* the file as shell code. Your
`.env` has spaces around `=` (`EE_PROJECT = value`), which shell assignment
forbids (Lecture 0.3), so it crashed. The fix reads the file as *text*:

```bash
EE_PROJECT=$(sed -n 's/^EE_PROJECT[[:space:]]*=[[:space:]]*//p' .env | head -1 | tr -d '"'"'"' \t\r')
```

`sed` is a stream editor; the expression `s/pattern//p` deletes the matched
prefix and prints the remainder, and `[[:space:]]*` tolerates any spacing
around `=`. `head -1` keeps the first match; `tr -d` strips quotes and stray
whitespace. General lesson: **never execute configuration; parse it.** Then
`: "${EE_PROJECT:?message}"` — parameter expansion with `:?` aborts with the
message if the variable is empty: fail *now*, at launch with a clear sentence,
not at 2 a.m. with a Python traceback.

### 6.4 nohup, `&`, and process lifetimes

When you close a terminal, the shell sends its child processes the **SIGHUP**
("hang up" — vocabulary from modems) signal, which normally kills them. Three
pieces make the job outlive the window: the trailing `&` runs it in the
**background** (shell prompt returns immediately); `nohup` makes it immune to
SIGHUP; the redirect `>> "$LOG" 2>&1` (append; stderr joined to stdout,
Lecture 0.2) gives its output somewhere to live. `$!` captures the **PID**
(process ID) of the job just launched. Inside the job, `&&` chains trials →
analysis → the ALL DONE echo, so the analysis only runs on a successful night
(Lecture 5.6's exit code, cashing out).

### 6.5 caffeinate: keeping the Mac honest

macOS idle-sleeps, which suspends the job. `caffeinate -im -w "$JOB_PID"`
prevents idle-sleep (`-i`) and disk-sleep (`-m`) *while process JOB_PID is
alive* (`-w` = watch that PID) — and exits by itself when the job ends, so the
machine can sleep once the science is done. The original advice to *wrap* the
script in caffeinate was my design bug: the wrapper script exits milliseconds
after launching the background job, and caffeinate — watching the wrapper —
exited with it. Watching the *job's* PID is the correct binding. Physical
caveat: lid-close overrides caffeinate; hence plugged in, lid open.

To watch the night: `tail -f data/timeseries/nohup_overnight.log` — `tail -f`
means *follow*: print new lines as they are appended. Ctrl-C stops watching,
not the job (they are unrelated processes).

---

## Lecture 7 — `analysis.py`: the statistics

### 7.1 Architecture: sections as independent experiments

One loader, one `Report` accumulator (a class that collects markdown lines and
writes them at the end — so a partial dataset still yields a readable report),
and eight sections, each wrapped in `try/except` in `main()` so a section that
cannot run (smoke-test data, say) degrades to a note instead of killing the
rest. `matplotlib.use("Agg")` selects the no-display backend — figures render
to PNG files on a machine where no window could open (a nohup job has no
screen). Constants (`TAU_OBSERVED = 0.95`, `T_MAIN = 0.4` …) sit at the top:
every analytic choice is one visible, greppable, criticizable number, not a
magic literal buried in a formula.

Two workhorse idioms recur. `pivot_table(index="chip_id", columns="period",
values="prob")` reshapes the long panel (one row per chip-period) into a wide
matrix (chips × periods) — the natural shape for "compare a chip across
months". And `dissolved_area_km2` projects to `EPSG:32719` (UTM zone 19S,
where coordinates are meters — you cannot measure area in degrees) and takes
the **union** of the chip squares before measuring: chips overlap 50 % by
construction, so summing them would double-count by ~4×.

### 7.2 Section A — coverage & missingness

Per period: mean `clear_frac`, share of chips fully observed (≥ 0.95), share
fully missing. This is the *descriptive foundation*: how much data does a
monthly cadence actually deliver, and when? Expect a wet-season trough
(Nov–Mar). The chronic-gap histogram (how many of 12 months each chip is
observed) identifies places where a monthly series would be mostly
interpolation — for those chips, "monthly" is an illusion regardless of how
clever the model is.

### 7.3 Section B — the fill-check aggregated

The per-tile `fill0_mismatch` (Lecture 4.3) averaged over the night. One number
converts a card footnote from assumption to measurement.

### 7.4 Section C — naive vs cloud-aware area, and why both

Two series at each threshold: `area_naive` (all chips — what the raw pipeline
would report) and `area_observed` (cloud-flagged chips only). The *gap between
them* is the mechanical cloud artefact; plotting both makes the case for the
flag visually, in one figure. Summary numbers per cadence: the **CV**
(coefficient of variation, std/mean — scale-free volatility), mean absolute
month-to-month change, and **lag-1 autocorrelation** (correlation of the
series with itself shifted one period; high = smooth/persistent, near zero =
noise-dominated). With n = 12 these are descriptive — the code says so in its
own output, which is deliberate: numbers that travel into slides should carry
their caveats with them.

### 7.5 Section D — is the "signal" just clouds?

The critical self-test. Period level: correlate *first differences* (Δ = this
month minus last — differencing removes the stable level and isolates change)
of area against Δ of coverage share, with **Spearman's ρ** (correlation of
ranks — robust to outliers and nonlinearity, wise at n = 11) beside an OLS
slope for effect size. If cloud-aware Δarea still tracks Δcoverage, then
*which* chips are observable is shifting the composition of the series —
the balanced-panel remedy is stated right there in the report. Chip level: do
classification *flips* between consecutive months coincide with coverage
changes? **Mann-Whitney U** (a rank test: are values in one group
systematically larger, no normality assumed) compares |Δclear_frac| of
flippers vs stable chips — catching partial-cloud noise that sneaks past the
τ = 0.95 filter.

### 7.6 Section E — do the two cadences even agree?

Restricted to chips fully observed in a quarter *and* all three of its months
— so disagreement cannot be blamed on missingness; what remains is temporal
aggregation, the median's non-linearity, and noise. Measures: Spearman ρ of
quarterly prob vs mean-of-monthly; **Cohen's κ** for the thresholded
classifications — agreement *corrected for chance* (two 95 %-negative raters
agree 90 % by luck alone; κ subtracts that; implemented in eight lines rather
than importing scikit-learn); and footprint **IoU** (intersection over union
of dissolved detection maps — position-sensitive where κ is only
label-sensitive). The report's reading guide pre-registers the interpretation:
`pos_rate_anyM > pos_rate_q` is the noise-accumulation signature ("any of 3
noisy looks" fires more often), while the reverse pattern would mean quarterly
composites are genuinely more sensitive — different conclusion, different
modeling advice, decided by data rather than post-hoc.

### 7.7 Section F — the identifying assumption, used and doubted

To say anything about *noise* without ground truth we need an assumption:
true mining at one 480 m chip changes little within a quarter, so the spread
(sd) of its three monthly probabilities is mostly measurement noise. The code
states this **identifying assumption** openly, notes where it fails (frontier
expansion is real within-quarter change — so the estimate is an *upper*
bound), and stratifies by quarterly-confident chips (< 0.1 or > 0.9): for a
chip the quarterly composite is *sure* about, monthly wobble is almost surely
pure noise. `flip_rate_within_q` then counts chips whose thresholded label is
inconstant within a quarter. The typical finding this design can support:
noise concentrates near the decision boundary — which is precisely the
argument for carrying the continuous probability into the econometrics and
thresholding last (Lecture 2.3).

### 7.8 Sections G, H — sensitivity and the precision mechanism

G varies the threshold over 0.30–0.60 and plots area against it: a steep curve
means the headline number is an artifact of an arbitrary constant; the
share-near-threshold statistic compares how much probability mass each cadence
parks within ±0.05 of t. H tests the *mechanism* behind monthly noise: the
median over k scenes gets more precise as k grows, so among
quarterly-confirmed negatives, monthly false-positive tendency (p95 of prob;
share above t) should *fall* as `n_clear` rises. If it does, `n_clear` earns
its place as an observation-error weight in the interpolation model — a
smooth precision weight beats a binary keep/drop rule.

### 7.9 Sections I, J — the payoff and the confessions

I translates all of the above into four instructions for the econometric
stage (never read low-score-under-cloud as zero; model the seasonal MNAR
missingness — **MNAR** = "missing not at random": the *cause* of missingness
(wet season) is correlated with the *thing measured* (mining activity), which
is exactly when naive interpolation biases you; use continuous probabilities;
difference on balanced panels). J lists the limitations in write-up-ready
form: one AOI chosen non-randomly, instrument ≠ ground truth, n = 12,
"quarterly as truth" is convenience, revisit aliasing. In papers as in code:
declare your assumptions where they are used.

---

## Lecture 8 — The session changelog (what was actually done, and why)

**Repository hygiene.** Fetched upstream; established that PR #29 merged
`2026dense_embed` into their `main` (their active branch going forward).
Audited your fork: `core/` byte-identical after syncing a 13-line-stale
`core/README.md`; models and configs current; recovered the missing
`..._config-t0.43.txt` for your exact ensemble; confirmed the ~430 other
upstream-only files are your intentional archives (old models, website docs,
retraining data). Found and archived the legacy `gee/` folder (deleted
upstream, older code locally) to `_archive/gee_pre-core_2026-07-27/` — the
precise "reappearing archive" blind spot you predicted, closed.

**Documentation truthing.** `pipeline_spatial_card.md`: added the merge note,
upgraded the area-unit quirk to *verified* (the log's "ha" is really km²), and
rewrote §5 statuses. `CODE_WALKTHROUGH.md`: status banner separating
legacy-SSL4EO sections from current reality; operational sections 0.5/0.6/0.8
rewritten for the CNN model, `core/` paths, and t = 0.4. New `GIT_PLAYBOOK.md`
codifying the fetch → diff → path-checkout routine and the visibility rules.

**New scientific machinery (`timeseries/`, ~1,100 lines, all outside
upstream folders).** `coverage.py` (Lecture 3), `panel.py` (Lecture 4),
`run_trials.py` (Lecture 5), `run_overnight.sh` (Lecture 6, including the
.env-parsing and caffeinate fixes you triggered), `analysis.py` (Lecture 7),
`explore_results.ipynb` (thin interactive layer), plus the
`la_pampa_corridor.geojson` AOI and `.gitignore` entries for the caches.

**Verification performed before handing over.** Byte-compiled every module;
proved `chip_windows` order-identical to upstream's `cut_chips` on three
geometries; ran the *entire* analysis on a synthetic 300-chip × 16-period
panel with known properties (seasonal gaps, threshold noise, planted
mid-year growth) and confirmed every section recovered what was planted, with
zero failures; checked the `InferenceEngine` constructor signature against
every call we make. What could not be verified from here — Earth Engine
authentication and the first real fetch — is exactly what your two-minute
smoke test covers.

**Epistemics of the night ahead.** The overnight run is an instrument
calibration, not a result: it measures the measuring device (monthly vs
quarterly cadence) on terrain chosen to be favorable. Its output decides the
*design* of the real phase-one time series — cadence, gating, weighting —
before any basin-scale spending. That ordering (calibrate, then scale) is the
project plan working as intended.

*— end of course —*
