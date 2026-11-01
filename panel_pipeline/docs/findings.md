# What we measured

Every number here comes from a real run or benchmark, with its source. Estimates are marked as
such. Collected on 23 Oct 2026 from the project's notes and threads; add new measurements here
rather than in a new file.

Contents:
1. Test runs: does the pipeline work, and what do the quality variables say?
2. Our scores against AMW's
3. The last day of each period
4. Speed: where the time goes
5. Earth Engine cost and limits
6. Overlaps in the chip and tile structure
7. Size on disk
8. The first cluster run, assembled (aca_full_2026-10-23)

---

## 1. Test runs

**test_region_2026-10-19** (subregion 2, UTM zone 20, 36 tiles, 19,044 chips; 2026 monthly
Jan–Sep and quarterly Q1–Q3; 19 Oct). Full tables: `history/reading_the_notebook_test_region_2026.md`.

- All 12 tasks complete, 36/36 tiles, no failures. Every integrity check passes: each planned
  chip present once, nothing unplanned, `confidence` = mean of the 6 votes, `n_clear ≤ n_obs`,
  no missing values.
- **Months add up to quarters exactly:** for all 57,132 chip-quarters, the quarter's `n_obs` and
  `n_clear` equal the sum of its three months (maximum difference 0). So the period windows meet
  with no gap or overlap.
- May 2026 is the cloudy month: mean `n_clear` 1.2, 37% of chips below one clear image, 22% of
  pixels blank. Other months have about 3–6 clear images; quarters 7–14.
- No chip with `n_clear < 1` was detected (> 0.4) in any period: missing data doesn't create
  detections here, but may hide some.
- Quarter vs its months: 316 chips detected in both, 74 only in a month (mostly one-month
  flickers), 19 only in the quarter (cleaner composite).

**test_region_para_2026-10-19** (subregion 4, zone 21, 42 tiles, 22,218 chips; 2026Q1–Q3):
detections > 0.4 were 56 / 732 / 787.

**amw_compare_s3_2026-10-19** (all of subregion 3, 34,825 tiles, 18.4 M chips, 2026Q3; 20–22
Oct on Ruby's Mac): complete, 0 failed tiles. The first part checked: 529 chips per tile, unique
chip_ids, `n_clear` 6.2–91 (mean 21).

**assemble_period.py** on both small runs (21 Oct): the postprocess flags equal AMW's own
`dual_threshold_filter` chip for chip in all 24 test periods.

## 2. Our scores against AMW's

Sources: `../../amw_comparison/referee_audit.md` (independent re-check, geometry-matched) and
`../../amw_comparison/compare_subregion3.ipynb` (23 Oct).

| run | quarter | ours > 0.4 | AMW | both | only ours | only AMW | scores identical |
|---|---|---|---|---|---|---|---|
| test_region (s2) | 2026Q1 | 84 | 84 | 84 | 0 | 0 | 100% |
| test_region (s2) | 2026Q2 | 105 | 105 | 105 | 0 | 0 | 100% |
| test_region (s2) | 2026Q3 | 146 | 148 | 145 | 1 | 3 | 0% (largest gap 0.074) |
| Pará (s4) | 2026Q1 | 56 | 27 | 26 | 30 | 1 | 8% (largest gap 0.27) |
| Pará (s4) | 2026Q2 | 732 | 732 | 732 | 0 | 0 | 100% |
| Pará (s4) | 2026Q3 | 787 | 787 | 786 | 1 | 1 | 67% |
| **all of subregion 3** | **2026Q3** | **25,841** | **25,671** | **25,440** | **401** | **231** | 34% (median gap 0.0013) |

- Where the imagery is the same, our code reproduces AMW's scores to ~1e-6 (Q2 everywhere).
- Every AMW detection in subregion 3 lands on one of our planned chips: the chip_id join holds
  at full size.
- Subregion 3, Q3: 583 of the 632 disagreements have our score between 0.35 and 0.45. They sit
  in two areas (around 55°W 4.5°N and 46°W 3°S); elsewhere scores are identical, and the 462
  border tiles have no disagreements. That pattern fits the last day (§3); inferred from the
  map, the scenes themselves were not checked.

## 3. The last day of each period

AMW's driver passes a quarter's last day (31 Mar, 30 Jun, 30 Sep, 31 Dec) as Earth Engine's
end date, which `filterDate` excludes, so that day is never imaged. Ours ends on the 1st of the
next period, so every day is in (STATE.md §4; Rmd D9).

*Measured* (`../../amw_comparison/last_day_check.ipynb`, 20 Oct): Pará 2026Q1, 56 detections
ours against 27 AMW's, and **the whole gap is 31 March**. Running AMW's own code with the two
end dates reproduces both results exactly. On Q1 and Q2 of test_region the last day added
nothing (no clear scene that day). Ruby planned to tell AMW; evidence to send: that rerun and
their file names ending `_2026-03-31`.

## 4. Speed: where the time goes

Measured on Ruby's MacBook (Apple M5), subregion 3, 2026Q3, 20–22 Oct (Live run diagnostics).

| What | Measured |
|---|---|
| Scoring, AMW's Mac setting (eager) | 2.95 s per tile |
| Scoring, compiled ("graph", now the default) | **1.85 s per tile**; scores differ by ≤ 1.2e-7, no chip crosses 0.4 |
| Scoring, "direct" (not used) | 1.71 s per tile, but differs from both AMW setups |
| Download size, 64-bit (before) | 34.5 MB per tile |
| Download size, 32-bit (now) | **18.6 MB per tile**, identical values |
| Throughput on Wi-Fi | ~20–23 tiles/min at 18.6 MB (10–12 at 34.5 MB): the Wi-Fi ceiling of about 60 Mbit/s. Halving the tile doubled the tiles |
| Workers | 16–22 workers, 0 Earth Engine retries; "scoring busy" 15–49%, so downloads were the limit |
| Old 300-tile batches | 16 min 38 s for the first batch (3.3 s per tile) |
| Laptop lid closed | the run froze; downloads hung on dead connections until the 300 s timeout was added |

On a Mac, only `--workers` adds speed: several processes share the same Wi-Fi.

## 5. Earth Engine cost and limits

From `checks/bench_download.py` on 5 tiles of subregion 3, 2026Q3 (22 Oct), and Google's
documentation (checked 23 Oct; see `cluster/cluster_plan_goethe.md` R2).

- **~130 EECU-seconds per tile-quarter** (56–213 across tiles); about 35 s per request on its own.
- 95 of 116 EECU-seconds in one profile are `reduce.median`: the per-pixel median over all clear
  observations. Earth Engine already loads only the images touching the tile (about 2,500
  Sentinel-2 assets in both recipes).
- Restricting the recipe to the tile (`filterBounds`): identical pixels on all 5 tiles, but no
  reliable saving (58 vs 110 EECU-s on one tile, the reverse on another). Not used.
- A monthly composite's cost is **not measured yet** (fewer images, probably cheaper).
- Limits: 40 concurrent requests per project by default (adjustable); Partner tier 100,000
  EECU-hours a month, after which the project is slowed ("restricted mode"), not cut off.
- *Estimate:* both limits meet at roughly 90–100k tile-periods a day. The full plan (38.5 M
  tile-periods) would take about 430 days; quarterly only about 105.

## 6. Overlaps in the chip and tile structure

Source: `open_design_decisions.Rmd` D5 and D6 (measured 19 Oct with `../checks/overlap/`).

| Level | Overlap |
|---|---|
| Subregions 1–6 | none |
| Tiles on subregion borders | 1,186 of 274,932 tiles listed twice (same key, same chip_ids) |
| Neighbouring tiles (padding) | a 24 px shared strip, median 4.2% of a tile per pair |
| UTM zone seams | a ~9 km strip at each seam scored twice on two unrelated grids; different keys, so no dedupe catches them |
| Chips inside a tile | ×4 (stride 24 on 48 px) |
| "Every other chip" | not exactly disjoint: ~0.7% of area double-counted at tile edges (the 8 Oct "exact 480 m grid" claim is corrected) |

Chip area: 231,666–233,666 m², not 230,400, because the pixels sit on a lat/lon grid.

## 7. Size on disk

- Detect output: 45.6 bytes per chip-period on disk (test_region). *Estimate* for all of
  Amazon_ACA: about 690 GB monthly and 230 GB quarterly.
- A test-run image cache: about 4 MB per tile-period (432 images = 1.7 GB).
- A full-ACA `plan.py`: about 2–3 h and 1.4 GB (*estimate*, 19 Oct).

## 8. The first cluster run, assembled (aca_full_2026-10-23)

Measured 1 Nov 2026 on the run's files (copied from Goethe), with `../assemble_period.py` and
`../checks/check_panel.py`.

**The run.** 16 tasks: subregions 2 and 3, each month of 2025-01 to 2025-06 and 2025Q1–Q2.
All complete, 0 failed tiles, on Goethe 25–29 Oct (365 task-hours). Subregion 2: 53,084 tiles,
28,081,436 chips; subregion 3: 34,825 tiles, 18,422,425 chips. Every complete task folder
holds a single `part_00000.parquet`: detect.py merges the parts when a task completes. Its
rows are not sorted by chip_id (about 865 row groups with overlapping ranges); assembly sorts.

**Border tiles.** 218 tiles (115,322 chips) are in both subregions. In all 8 periods the two
copies are identical in every column, scores and quality alike.

**Masked pixels.** Since the 32-bit downloads (22 Oct) a pixel with no clear image arrives as
`-inf`, and detect.py took it for a seen pixel (fixed 1 Nov, README A6). Chips with such a
pixel, by task:

| Period | Subregion 2 | Subregion 3 |
|---|---|---|
| 2025-01 | 29.6% | 58.4% |
| 2025-02 | 63.8% | 80.2% |
| 2025-03 | 36.6% | 56.7% |
| 2025-04 | 36.1% | 41.6% |
| 2025-05 | 47.5% | 22.8% |
| 2025-06 | 33.9% | 21.6% |
| 2025Q1 | 7.2% | 21.7% |
| 2025Q2 | 5.8% | 0.8% |

Up to 26% of chips have no clear image at all (`n_clear = 0`, subregion 3, February).

**Same AMW centre, different chip.** At the UTM zone seam at 60°W (zones 20 and 21), 368 chip
pairs of subregion 2 share AMW's rounded centre. AMW keeps one of each pair before the
neighbour search; so do we (Rmd D5).

**Assembly against AMW.** Two samples of the run: 947,968 chips (including 32 border tiles)
and 379,822 chips across the 60°W seam. In all 8 periods the flags and distances equal those
of AMW's own `dual_threshold_filter` run on the same scores, also with `--subregions 3`. The
two old test runs re-assembled give identical values, and every check passes. Deliberately
broken inputs are caught: differing border copies (the higher is kept and the border check
fails), a chip twice in one subregion and a missing tile (assembly refuses), a flag changed
by hand (the flags check fails).

**What the panel shows** (the first sample):
- 2025Q2: 12,876 raw detections, 11,656 loose, 8,257 stringent.
- A chip with no clear image (a black square) scores at most 0.000.
- In the months, raw detections per 1,000 chips rise with `n_clear`: 0.7 below 1, 7.7 from
  1 to 3, 12.2 from 3 to 10. A monthly count mixes mining with the weather (Rmd D1, D8).
- Of the chips detected in a quarter, 80–87% are also detected in at least one of its months.

**Speed and memory.** A period of 10.5 M chips (a test run made for this): 40 s, 2.0 GB.
*Estimate* for subregions 2 and 3 (46.4 M chips): about 3 min and 8 GB per period.
