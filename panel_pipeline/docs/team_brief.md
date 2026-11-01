# Panel Pipeline: Team Brief

Oct 19, 2026, updated Oct 23 · @Ruby

## Where things live

We score every 48 px chip of the Amazon with Amazon Mining Watch's (AMW) trained model, month by month and quarter by quarter from 2018, and write the scores as a chip × period panel for econometric work. We do not publish anything, so we use AMW's tiling and inference code and leave the rest of their pipeline alone.

| What | Where |
| --- | --- |
| Repository | [RubyAnn777/AMW-mining-detector](https://github.com/RubyAnn777/AMW-mining-detector), branch `pipeline-rebuild` |
| Our pipeline | `panel_pipeline/` (every edit we made is in this folder) |
| AMW's pipeline | `core/` (their library) and `scripts/` (their driver, `scripts/pipeline.py`); we change nothing there |
| Model | `models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` (AMW's) |
| Region | `data/boundaries/Amazon_ACA/Amazon_ACA_1…6.geojson`, plus `andes_supplemental.geojson` and the test area `test_region_small.geojson` |
| Settings | one file per run in `configs/`, copied from the template `panel_pipeline/config.yaml` |
| Test runs | `tests_output/test_region_2026-10-19` and `tests_output/test_region_large_2026-10-19`, configs in `configs/`; `tests_output/` is not in git |
| Notebook | `panel_pipeline/checks/explore_runs.ipynb` |
| Documents | `panel_pipeline/docs/` (start at `docs/README.md`); how to run: `panel_pipeline/README.md` |

The old `data/timeseries*` attempt is not used by anything here.

## What we take from AMW

We import four AMW files from `core/` and nothing else; they run unchanged. `lib/grid.py` puts `core/` on Python's search path, so these files must keep their names and location.

| AMW file | What we use | Where we use it | How |
| --- | --- | --- | --- |
| `core/tile_utils.py` | `create_tiles`, `cut_chips` | `grid.py`, `detect.py` | Unchanged: AMW's tile grid, and cutting each tile into 48 px chips |
| `core/gee.py` | `GEE_Data_Extractor`, `DataConfig`, the band list, the Earth Engine login | `detect.py` | Subclassed as `PanelExtractor`: AMW's cloud-masked median composite and download are kept; we add two count bands to the same request and make its optional image cache write safely |
| `core/postprocess.py` | `centroid_key`, and its dual-threshold and neighbour functions | `grid.py`, `assemble_period.py` | Unchanged: AMW's rounded-centroid key, stored next to our `chip_id` so our rows can be matched to AMW's files |
| `core/tf_darwin.py` | `apply_darwin_tf_compat` | `detect.py` | Unchanged, macOS only, and only with `--mac-eager` (by default a Mac scores the same compiled way as Linux) |

We also copy AMW's scoring arithmetic line for line in `detect.py` (bands ÷ 10000, clipped to 0–1; chip stride 24 px; confidence = mean of the ensemble's votes), and load AMW's model file as is. Every reuse or change is marked in our code as `# AMW: file:line`, `# Same as AMW`, `# DEVIATION from AMW` or `# ADDITION`; `grep -rn "DEVIATION\|ADDITION" panel_pipeline/` lists all of them. One quirk: `postprocess.py` needs scikit-learn, which AMW's `requirements.txt` leaves out.

## Our files

Four scripts are run by hand (`plan.py`, `detect.py`, `status.py`, `assemble_period.py`) and sit at the top of `panel_pipeline/`; the helpers they import are in `lib/`, and checks and benchmarks in `checks/`.

| File | What it does | Run it? |
| --- | --- | --- |
| `config.yaml` | Every setting of a run: name and output folder, Earth Engine project, months and quarters (2018-01 to 2026-09), subregions and test clip, model, imagery, thresholds | Edit |
| `lib/config.py` | Reads `config.yaml` and refuses typos, missing keys, impossible values and keys written twice; `python panel_pipeline/lib/config.py` prints a summary | Optional check |
| `lib/periods.py` | A period is a month (`2024-01`) or a quarter (`2024Q1`), with its start date and its end date (the first day of the next period) | Helper |
| `lib/grid.py` | AMW's tiles over a region, every chip in each tile, and our whole-number `chip_id` | Helper |
| `plan.py` | Makes the run folder once: frozen copy of the settings, `provenance.json`, `plan/tiles.csv` (tiles per subregion) and `plan/chips.parquet` (every chip with its square, AMW key, `in_region` and `in_andes` flags) | Yes, once per run |
| `detect.py` | Downloads each tile's composite from Earth Engine, scores every chip with AMW's model, and saves one Parquet folder per period and subregion: a part every 2 minutes, merged into one file when the task is done; `_log.txt` with speed and errors; Ctrl-C saves; resumable | Yes, one task = frequency × period × subregion |
| `lib/lock.py` | One writer per period folder: a `_lock.json` with a heartbeat, so two processes never write the same folder | Helper |
| `lib/provenance.py` | Records code version, uncommitted files, package versions and a fingerprint of the model and boundary files | Helper |
| `status.py` | One line per year and one mark per task (done, running, failed, partly done, not started); `--todo` prints the commands still to run | Yes, any time |
| `assemble_period.py` | One period across all subregions: checks every task is complete, then writes the panel layers `scores` (every planned chip, failed tiles as missing) and `postprocess` (AMW's loose and stringent flags and distances) | Yes, per frequency × period |
| `lib/read_panel.py` | Reads the panel layers side by side | Helper |
| `checks/explore_runs.ipynb` | Read-only look at a run: folder layout, settings, `chip_id` check, task list, variables and summary stats, monthly vs quarterly join and count check, combining runs, maps, full-run size estimate | Open in Jupyter |

Each row of the output is one chip in one period: `chip_id`, `confidence` (AMW's score), `vote_1`…`vote_6` (each ensemble member), `n_obs` and `n_clear` (images per pixel, all and cloud-free) and `empty_share` (share of pixels with no clear image).

## How it was run

The test run `test_region` covers 36 tiles (19,044 chips, all in subregion 2, all in UTM zone 20) and was scored on a laptop for January to September 2026, monthly and quarterly. Every period finished with 36 of 36 tiles and no failures, about 2 minutes per quarter, and each quarter's image counts equal the sum of its three months for every chip. A larger plan, `test_region_large`, is set up for the next comparison. Steps, from the repository root:

1. Write the run's settings in a config file under `configs/` (`run.name`, `output_root: tests_output`, `clip_to` set to the test area).
2. `python panel_pipeline/plan.py --config configs/<file>.yaml` made `tests_output/test_region_2026-10-19/`.
3. `python panel_pipeline/detect.py --run tests_output/test_region_2026-10-19 --frequency monthly --subregion 2 --from 2026-01 --to 2026-09 --image-cache tests_output/image_cache`, then the same with `--frequency quarterly --from 2026Q1 --to 2026Q3`. The image cache is for test runs only.
4. `python panel_pipeline/status.py --run tests_output/test_region_2026-10-19` shows what is done.
5. `checks/explore_runs.ipynb` reads the run (its `RUNS` cell points at it; uncomment the second line to compare with `test_region_large`).

The output is laid out as `<run>/detect/<frequency>/period=<tag>/subregion=<n>/part_00001.parquet`, which pandas, DuckDB, R `arrow` read as one table with `period` and `subregion` columns. Monthly and quarterly share the same `chip_id`s, so they join directly. Run outputs are not in git; ask Ruby for the folder.

## How we differ from AMW

A chip's score is computed exactly as AMW computes it: same model, same Sentinel-2 median composite and cloud test (Cloud Score+ ≥ 0.6), same tile grid, same chips, same arithmetic. What changes is what we keep, how it is organised and how runs survive failures.

| Kind | AMW | Ours | Why |
| --- | --- | --- | --- |
| Structure | `scripts/pipeline.py` drives 13 stages, from review-config to publish; most only print commands a person runs in tmux on VMs | Four scripts: `plan.py`, `detect.py`, `status.py`, `assemble_period.py`; one task = one period × subregion folder, ready for a cluster | We need scores, not a website |
| Structure | Stages after inference: concat, filter, postprocess, persistence, SAM2 masks, map rasters, staging, manifest, publish | Concat and postprocess kept, as flags in `assemble_period.py`; persistence will be a flag too; nothing is filtered out or deleted; SAM2 is deferred | Keep every score; decide rules in the analysis |
| Structure | Settings spread over constants and defaults in several files | One checked `config.yaml`, frozen into the run folder at planning | One place, and a half-finished run can't change |
| Periods | Calendar years, quarters from 2025; tags `2024`, `Q125` | Months and quarters, 2018-01 to 2026-09 (105 + 35); tags `2024-01`, `2024Q1`; never mixed | Panel frequency; tags sort and parse in pandas |
| Periods | Current driver ends a period on its last day, which Earth Engine excludes | End is the first day of the next period | No day dropped; quarter = its three months exactly |
| Output | Only chips above 0.4 (Andes 0.2), as GeoJSON, votes as text | Every chip, every period, as Parquet: `confidence` (float64), 6 votes, quality counts | Below-threshold scores are data too |
| Chip identity | Rounded chip centre, matched across periods by coordinates | Whole-number `chip_id` from tile and chip position; AMW's key kept beside it | Exact join across periods, frequencies and runs |
| Andes | Second inference pass over the Andes at 0.2 | One pass; an `in_andes` flag; 0.2 applied later | Same scores, half the downloads |
| Quality | None | `n_obs`, `n_clear`, `empty_share`, added as bands to the same Earth Engine request | Controls for cloud cover with one call per tile |
| Region flag | Keeps every chip of a tile touching the region | Same, plus an `in_region` flag | Lets the analysis restrict the sample |
| Earth Engine project | Environment variable, default AMW's own | Set in `config.yaml` | Billing is explicit |
| `gee.py` | `GEE_Data_Extractor` | Subclass `PanelExtractor`: adds the two count bands; cache writes are all-or-nothing | Reuse AMW's download unchanged |
| Image cache | Every image cached on disk, then re-read | No cache; resume from saved results (`--image-cache` for tests only) | A full cache would be 200 to 300 TB, and cached images are rounded, changing scores slightly |
| Download loop | Starts all 500 downloads of a batch at once (about 9 GB in memory) | Downloads never pause, at most two per worker in flight, scored as they arrive; 32-bit pixels (18.6 MB a tile instead of 34.5) | Same scores, small memory, half the download |
| Failures | A rerun starts over; only the count of failed tiles is logged | Resumes from saved parts; `_status.json` lists failed tiles and why; folder locks; clean stop on Ctrl-C or a time limit | A long cluster run must restart safely |
| Provenance | Model, threshold and dates in file names | `provenance.json` per run and a record per session: code commit, packages, machine, file fingerprints | Every number traceable |
| Test runs | Point inference at another region file | `clip_to` cuts each subregion to the test area | Tests go through the same routing as the full run |

Same as AMW on purpose: tiles on a subregion border are listed under both subregions and scored twice. AMW removes the copies in its concat stage; ours carry the same `chip_id`, so one row per chip must be kept when the panel is assembled.

## What is left

Scoring works end to end; it ran on all of subregion 3 for 2026Q3 (34,825 tiles, none failed), and assembly works on the test runs. Details and every open question: `docs/STATE.md` and `docs/open_design_decisions.Rmd`.

- [x] **Compare with AMW for 2026.** Where the imagery is the same, our scores equal AMW's to ~1e-6. The remaining gaps come from AMW dropping each quarter's last day (31 Mar, 30 Jun, 30 Sep): proven on Pará 2026Q1 by rerunning AMW's own code. Subregion 3, Q3: 25,841 detections ours, 25,671 AMW's, 97.6% overlap. See `docs/findings.md`.
- [ ] **Move to the Goethe cluster.** Job scripts are in `cluster/`, the how-to in `panel_pipeline/README.md` Part B. The limit is Earth Engine, not the cluster: about 90k tile-periods a day, so the full plan (38.5 M) would take about 14 months. The scope of periods and the project's quota are being checked.
- [ ] **Settle the analysis unit.** Chips overlap 4× by design, and UTM zone seams are scored twice. Proposed: keep chips as the measurement and build 240 m cells that don't overlap for the analysis (Rmd D5, D6).
- [ ] **Persistence** as a flag, across time (`build_panel.py`); the rule is open (Rmd D8).
- [ ] **Later: SAM2 mine areas**, as a separate step that adds its own columns.
