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
├── download_amw.py   fetches AMW's files, only needed for quarters you don't have yet
├── compare.ipynb     the comparison: inputs, join, counts, scores, diagnostics, map
├── tests/            python -m unittest discover -s amw_comparison/tests
└── data/amw/         AMW's published files (ignored by git)

configs/amw_compare_s3_2026-10-19.yaml     the long run's settings (committed)
tests_output/amw_compare_s3_2026-10-19/    the long run's output (ignored by git)
```

How the join works: AMW's files have no chip ID, only each chip's square. AMW identifies
a chip by its centre rounded to 5 decimals (`centroid_key`, core/postprocess.py:113), and
`plan.py` already stores that key for every chip (`amw_key_x/y` in `plan/chips.parquet`).
Equal keys mean the same chip. Checked on AMW's real files: all 8,101 / 19,389 / 25,646
AMW detections centred in subregion 3 (Q1 / Q2 / Q3 2026) land on one of our planned chips.

## Results so far: test_region (36 tiles in subregion 2)

| quarter | ours | AMW | both | only ours | only AMW | scores identical |
|---|---|---|---|---|---|---|
| 2026Q1 | 84 | 84 | 84 | 0 | 0 | 100% |
| 2026Q2 | 105 | 105 | 105 | 0 | 0 | 100% |
| 2026Q3 | 146 | 148 | 145 | 1 | 3 | 0% (median gap 0.008, largest 0.074) |

- **Q1 and Q2 are reproduced exactly**: the same chips, and the same scores and six votes to
  about 1e-7 (rounding in the model's arithmetic).
- **Q3 differs slightly everywhere**: every tile, all six votes. That means the median was made
  of different images, not that the code differs. The 4 disagreements all have scores between
  0.35 and 0.45. Two possible causes:
  1. **30 September.** Our quarter ends on 1 Oct, excluded, so 30 Sep is in. AMW's ends on
     30 Sep, excluded (handoff §4). On Q1 and Q2 the last day added nothing here; in Q3 it may.
  2. **Scenes Earth Engine added after AMW scored Q3** (AMW scored it only weeks ago).
  `scene_dates.py` (step 3) tells them apart.

AMW files compared (sha256, first 12 characters): Q126 `b4f35df7398e`, Q226 `5c57a94daca9`,
Q326 `f2f56a0484f2`. AMW replaces its files when it re-runs, and the notebook prints these.

## Steps

All commands from the repository root, in the venv.

**0. Once: AMW's files into `data/amw/`.** Your downloads are in `AMW_Output/`, which git
tracks. This branch stops tracking it, so copy the files first; the pull then removes
`AMW_Output/` and the copy stays:

```bash
mkdir -p amw_comparison/data
cp -R AMW_Output amw_comparison/data/amw
git pull origin claude/project-thread-scl620
```

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
    --frequency quarterly --subregion 3 --from 2026Q2 --to 2026Q2 --workers 3 --batch-tiles 100
```

- Why subregion 3: the smallest (34,825 tiles, 18.4 M chips), no Andes, 462 border tiles.
- Why 2026Q2: it reproduced exactly on test_region, so any difference at scale is a real
  finding. Q3 can be added to the same run later (`--from 2026Q3 --to 2026Q3`).
- `plan.py` takes about 15 minutes and writes about 160 MB.
- `caffeinate -i` keeps the Mac awake while the command runs (plugged in, lid open).
- `--workers 3` is the low setting; `--batch-tiles 100` saves every 100 tiles, so a stop
  loses at most one batch. If it stops for any reason, run the same command again: it resumes.
- Progress: `python panel_pipeline/status.py --run tests_output/amw_compare_s3_2026-10-19`.

**Runtime.** Your test_region quarters took 1.7 to 2.3 minutes for 36 tiles with 6 workers,
about 3 to 4 seconds per tile. At that pace subregion 3 takes about 30 to 36 hours; with 3
workers up to twice that if Earth Engine is the bottleneck, so plan for **1.5 to 3 days**. The
log prints the minutes for each 100-tile batch; minutes per batch × 349 batches is the total.

**5. Compare.** In the notebook set `RUN = "tests_output/amw_compare_s3_2026-10-19"`,
`SUBREGION = 3`, Run All. Keeping only detected chips, it stays within a few GB of memory.
