# AMW comparison

Does our pipeline reproduce AMW's published quarterly detections? We compare, chip by chip,
our `detect.py` scores with AMW's raw detections (confidence > 0.4) of the same quarter: which
chips each side detects, and how close the scores are. Area is left out. AMW has no months, so
our three months are compared with AMW's quarter.

## Folder

```
amw_comparison/
├── compare.py      joins a run with AMW's files and writes <run>/amw_comparison/ (the results)
├── compare.ipynb   runs the join if needed, then shows it: counts, scores, last day, border, months
├── last_day.py     are the differences under the scenes of the quarter's last day? (Earth Engine)
├── tests/          python -m unittest discover -s amw_comparison/tests
└── data/amw/       AMW's files as you downloaded them (ignored by git)
```

**The join.** AMW's files have no chip ID, only each chip's square. AMW identifies a chip by its
centre rounded to 5 decimals (`centroid_key`, core/postprocess.py:113), and `plan.py` stores that
key for every chip (`amw_key_x/y` in `plan/chips.parquet`). Equal keys mean the same chip. Where
one location has two of our chips (tiles overlapping where UTM zones meet, and border tiles, which
both neighbouring subregions score), the higher score is kept, as AMW does
(`dedupe_detections`, core/postprocess.py:146).

## Steps

All commands from the repository root, in the venv.

**1. AMW's files, by hand.** From https://source.coop/earthgenome/amazon-mining-watch, folder
`raw_detections/`, the `_t0.4` file of each quarter, into `amw_comparison/data/amw/`:
`amazon_basin_Q125_t0.4.geojson` is 2025Q1, `amazon_basin_Q326_t0.4.geojson` 2026Q3.

**2. Compare.** Open `compare.ipynb`, set `RUN` and `QUARTERS` in the first cell, *Run All*.
The first time, section 1 runs the join and saves `<run>/amw_comparison/`; after that the
notebook takes seconds (`REDO = True` joins again). The quick check is the test region (36
tiles, under a minute); a whole-basin run needs its `plan/` and `detect/` folders on the laptop,
and takes about 2 minutes and 6.5 GB of memory per subregion and quarter (measured on
subregion 3, 18.4 M chips; the largest subregion has about 1.5 times as many).

The same join as a command, for example inside a job on Goethe (then copy only the small results
folder to the laptop and open the notebook there):

```bash
python amw_comparison/compare.py --run tests_output/aca_full_2026-10-23 --quarters 2025Q1 2025Q2
python amw_comparison/compare.py --run <run> --year 2026     # the year's quarters that have an AMW file
```

On Goethe, as a job, and the notebook as a web page (`panel_pipeline/README.md` B8):
`sbatch cluster/amw_compare.sbatch $RUN 2026`, then
`sbatch cluster/notebook.sbatch amw_comparison/compare.ipynb $RUN YEAR=2026`.
`inputs.csv`, `joins.csv` and `amw_files.csv` keep the rows of quarters compared earlier, so
comparing 2025 after 2026 adds to them.

**3. The last day** (needs Earth Engine): set `CHECK_LAST_DAY = True` in the notebook, or

```bash
python amw_comparison/last_day.py --run tests_output/aca_full_2026-10-23 --quarter 2025Q1
```

**4. The full last-day check** (needs Earth Engine and TensorFlow; on Goethe as a job):

```bash
python amw_comparison/last_day_check.py --run <run> --year 2026        # or --quarters 2026Q1
sbatch cluster/last_day_check.sbatch $RUN 2026                          # the same, on Goethe
```

Per quarter, in `<run>/amw_comparison/last_day_check/<quarter>/`: the last-day table per
subregion (`by_subregion.csv`), the disagreeing chips outside the last day's scenes with
their score gaps (`outside.csv`), and the decisive test (`rescore.csv`): a few tiles per
subregion with disagreements, plus one control tile, downloaded and scored again with
(a) AMW's own downloader and AMW's end date, which must give AMW's published scores,
(b) ours and our end date, which must give our scores, and (c) AMW's 64-bit downloader with
our end date, which must also give our scores (so the number format is not the difference).
`summary.txt` says what passed.

What `<run>/amw_comparison/` holds:

| file | what |
|---|---|
| `inputs.csv` | every task read (frequency, period, subregion): complete, tiles, failed tiles |
| `amw_files.csv` | AMW's files compared, rows and fingerprint (AMW replaces files when it re-runs) |
| `joins.csv` | per quarter and subregion: AMW rows on our chips, rows that should have matched (must be 0), our chips sharing a location |
| `chips_<quarter>.parquet` | every chip that our quarter, AMW or one of our months detected: `confidence`, `amw_confidence`, `conf_<month>`, `months_detected`, image quality, `subregion`, `border_copies`, `border_spread` |

## Results

**The last day explains the differences.** AMW's quarter ends a day early: its code passes the
quarter's last day as Earth Engine's end date, which is excluded, so 31 Mar, 30 Jun, 30 Sep and
31 Dec are never imaged (AMW's file names end `_2026-03-31`). Ours ends on the 1st of the next
month. We keep the last day: our quarters are whole calendar quarters, like our months.

- **Pará test area, 2026Q1** (20 Oct). We detect 56 chips, AMW 27. AMW's own code, run with AMW's
  end date, reproduces AMW's published file exactly (27 of 27, identical scores). Run with our
  end date, it reproduces our scores exactly (56 of 56). Our code is AMW's; only the date differs.
  Q1 is the rainy season (median 4.6 clear looks per pixel against 16.6 in Q2), so one extra
  clear scene moves the median image a lot.
- **Whole basin, 2025Q1** (30 Oct, subregions 2 and 3): 196 scenes taken on 31 March. All 104 chips
  only one side detects (44 ours, 60 AMW's) and 900 of the 904 chips both detect with different
  scores lie under those scenes. Elsewhere the scores are identical.

**Counts**

| run | quarter | ours | AMW | both | only ours | only AMW | scores identical |
|---|---|---|---|---|---|---|---|
| test region (s2, 36 tiles) | 2026Q1 | 84 | 84 | 84 | 0 | 0 | 100% |
| test region | 2026Q2 | 105 | 105 | 105 | 0 | 0 | 100% |
| test region | 2026Q3 | 146 | 148 | 145 | 1 | 3 | 0% (largest gap 0.074) |
| Pará (s4, 42 tiles) | 2026Q1 | 56 | 27 | 26 | 30 | 1 | 8% (largest gap 0.27) |
| Pará | 2026Q2 | 732 | 732 | 732 | 0 | 0 | 100% |
| subregion 3, all 34,825 tiles | 2026Q3 | 25,841 | 25,671 | 25,440 | 401 | 231 | 34% (median gap 0.0013) |

- Every AMW detection on our grid lands on one of our chips (0 "should have matched"), also at
  full size. One AMW chip in subregion 2, 2025Q1 falls between ours (a seam case, not checked).
- Disagreements sit near the threshold: in subregion 3, 583 of the 632 have our score between
  0.35 and 0.45.
- On 19 Oct the test region and Pará were re-checked independently, matching AMW's squares to
  ours by overlap instead of the key: the same pairs and the same numbers.

AMW files compared (sha256, first 12 characters): Q126 `b4f35df7398e`, Q226 `5c57a94daca9`,
Q326 `f2f56a0484f2`. The notebook prints the fingerprints of the files it used.

Upstream, checked 19 October (earthrise-media/mining-detector at `7bde733`): AMW's code that
scores a quarter (`core/gee.py`, `inference_engine.py`, `inference_pipeline.py`, `tile_utils.py`,
`periods.py`, `postprocess.py`) is the same as ours apart from comments.
