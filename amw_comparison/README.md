# AMW comparison

Does our pipeline reproduce AMW's published quarterly detections? We score the same
area and quarter with `panel_pipeline/detect.py` and compare chip by chip with AMW's raw
detections (confidence > 0.4): which chips each side detects, and how close the scores are.
Area is left out.

## Folder

```
amw_comparison/
├── README.md         this file
├── compare.py        reads our run and AMW's file, joins them on AMW's chip key, counts
├── scene_dates.py    lists the Sentinel-2 scenes that went into a quarter (explains gaps)
├── compare.ipynb     both tables' first rows, then inputs, join, counts, scores, diagnostics, map
├── tests/            python -m unittest discover -s amw_comparison/tests
└── data/amw/         AMW's files as you downloaded them (ignored by git)

configs/amw_compare_s3_2026-10-19.yaml     the long run's settings (committed)
tests_output/amw_compare_s3_2026-10-19/    the long run's output (ignored by git)
```

How the join works: AMW's files have no chip ID, only each chip's square. AMW identifies
a chip by its centre rounded to 5 decimals (`centroid_key`, core/postprocess.py:113), and
`plan.py` already stores that key for every chip (`amw_key_x/y` in `plan/chips.parquet`).
Equal keys mean the same chip. Checked on AMW's real files: all 8,101 / 19,389 / 25,646
AMW detections centred in subregion 3 (Q1 / Q2 / Q3 2026) land on one of our planned chips.
AMW's 16 October correction of a grid misalignment in subregion 3 concerns its 2025 annual
detections only (their public README); the quarterly files line up, as this check shows.

## Results

**Subregion 3, 2026Q3** (all 34,825 tiles, scored 20 to 22 October; `compare.ipynb` set to this run):

| ours | AMW | both | only ours | only AMW | overlap | scores identical | median gap | largest gap |
|---|---|---|---|---|---|---|---|---|
| 25,841 | 25,671 | 25,440 | 401 | 231 | 0.976 | 34% | 0.0013 | 0.23 |

- Every AMW detection in subregion 3 lands on one of our chips (0 "should have matched").
- 583 of the 632 disagreements have our score between 0.35 and 0.45.
- The differences are in two areas (around 55°W 4.5°N and 46°W 3°S); elsewhere the scores are
  identical. That fits the last day: 30 Sep is in our quarter and not in AMW's, and only the
  areas a clear 30 Sep scene covered change.
- No disagreements on the 462 border tiles.

**Why quarters differ: the last day** (settled, see `referee_audit.md` and
`last_day_check.ipynb`). AMW's quarter ends a day earlier (31 Mar, 30 Jun, 30 Sep are dropped).
Rerunning AMW's own code on Pará 2026Q1 with our end date reproduces our scores exactly, and with
AMW's end date reproduces AMW's published file exactly. We keep the last day.

**test_region** (36 tiles in subregion 2), for reference:

| quarter | ours | AMW | both | only ours | only AMW | scores identical |
|---|---|---|---|---|---|---|
| 2026Q1 | 84 | 84 | 84 | 0 | 0 | 100% |
| 2026Q2 | 105 | 105 | 105 | 0 | 0 | 100% |
| 2026Q3 | 146 | 148 | 145 | 1 | 3 | 0% (median gap 0.008, largest 0.074) |

Upstream, checked 19 October (earthrise-media/mining-detector at `7bde733`): AMW's code that
scores a quarter (`core/gee.py`, `inference_engine.py`, `inference_pipeline.py`, `tile_utils.py`,
`periods.py`, `postprocess.py`) is the same as ours apart from comments, and the date windows
and inference commands for Q3 are unchanged. Their newer commits are about publishing, area
statistics and docs.

AMW files compared (sha256, first 12 characters): Q126 `b4f35df7398e`, Q226 `5c57a94daca9`,
Q326 `f2f56a0484f2`. AMW replaces its files when it re-runs, and the notebook prints these.

## Steps

All commands from the repository root, in the venv.

**0. AMW's files.** The notebook reads them from `amw_comparison/data/amw/`, names as
downloaded from Source Cooperative:

```
amazon_basin_Q126_t0.4.geojson   amazon_basin_Q126_t0.43_t-iso0.75.geojson
amazon_basin_Q226_t0.4.geojson   amazon_basin_Q226_t0.43_t-iso0.75.geojson
amazon_basin_Q326_t0.4.geojson   amazon_basin_Q326_t0.43_t-iso0.75.geojson
```

The comparison needs only the `_t0.4` files; the others are shown in section 0. A later
quarter needs its files added there by hand. If `amazon_basin_detections.geojson` (AMW's one
row per location) is there too, section 0 shows it.

**1. Tests** (seconds, no Earth Engine): `python -m unittest discover -s amw_comparison/tests -v`

**2. Quick check on test_region** (already scored): open `compare.ipynb`, keep
`RUN = "tests_output/test_region_2026-10-19"`, Run All. Takes under a minute.

**3. Why Q3 differs** (metadata only, a few small Earth Engine requests):

```bash
python amw_comparison/scene_dates.py --run tests_output/test_region_2026-10-19 --quarter 2026Q3
```

It lists the scenes of 30 Sep (ours only) and when Earth Engine ingested each late-September
scene. Run it for 2026Q1 and 2026Q2 too: if those show no clear scene on their last day, the
last day explains everything. Then set `CHECK_LAST_DAY = True` in the notebook to draw the last
day's scenes on the map.

**4. The long run: all of subregion 3, quarter 2026Q2.**

```bash
python panel_pipeline/plan.py --config configs/amw_compare_s3_2026-10-19.yaml
caffeinate -i python panel_pipeline/detect.py --run tests_output/amw_compare_s3_2026-10-19 \
    --frequency quarterly --subregion 3 --from 2026Q3 --to 2026Q3 --workers 3 --batch-tiles 100
```

- Why subregion 3: the smallest (34,825 tiles, 18.4 M chips), no Andes, 462 border tiles.
- The run that was done is 2026Q3. Another quarter goes into the same run with `--from`/`--to`.
- `plan.py` takes about 15 minutes and writes about 160 MB.
- `caffeinate -i` keeps the Mac awake while the command runs (plugged in, lid open).
- `--workers 3` is the low setting; `--batch-tiles 100` saves every 100 tiles, so a stop
  loses at most one batch. If it stops for any reason, run the same command again: it resumes.
- Progress: `python panel_pipeline/status.py --run tests_output/amw_compare_s3_2026-10-19`.

**Runtime.** Your test_region quarters took 1.7 to 2.3 minutes for 36 tiles with 6 workers,
about 3 to 4 seconds per tile. At that pace subregion 3 takes about 30 to 36 hours; with 3
workers up to twice that if Earth Engine is the bottleneck, so plan for **1.5 to 3 days**. The
log prints the minutes for each 100-tile batch; minutes per batch × 349 batches is the total.

**5. Compare.** Open `compare.ipynb`, point it at this run and 2026Q3, and Run All. It takes about 2 minutes and 5 to 6 GB of memory.
