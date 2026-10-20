# Referee audit of the AMW comparison (19 Oct 2026)

Independent re-check of `amw_comparison/` (branch `claude/project-thread-scl620`) against the
files on Ruby's Mac: `tests_output/test_region_2026-10-19` (subregion 2, 36 tiles),
`tests_output/test_region_para_2026-10-19` (subregion 4, 42 tiles) and AMW's three `_t0.4` files.
The scripts are `independent_check.py`, `tile_breakdown.py` and `compare_rerun.py` in this folder. They match AMW rows to
our chips by **geometry** (overlap of the squares), not by compare.py's rounded-centre key, and
count every AMW row inside our footprint, so they don't share compare.py's assumptions.

## Reproduced numbers

| run | quarter | ours >0.4 | AMW | both | only ours | only AMW | scores identical (gap ≤1e-6) | largest gap |
|---|---|---|---|---|---|---|---|---|
| test_region (s2) | 2026Q1 | 84 | 84 | 84 | 0 | 0 | 100% | 4e-7 |
| test_region (s2) | 2026Q2 | 105 | 105 | 105 | 0 | 0 | 100% | 2e-7 |
| test_region (s2) | 2026Q3 | 146 | 148 | 145 | 1 | 3 | 0% | 0.074 |
| **Pará (s4)** | **2026Q1** | **56** | **27** | **26** | **30** | **1** | **8%** | **0.27** |
| Pará (s4) | 2026Q2 | 732 | 732 | 732 | 0 | 0 | 100% | 1.3e-6 |
| Pará (s4) | 2026Q3 | 787 | 787 | 786 | 1 | 1 | 67% | 0.076 |

## Findings

1. **"Q1 and Q2 match exactly" is not true in general.** It holds for the one 36-tile
   test_region. In the Pará run (scored 12:34–12:43 UTC, never reported in the comparison thread
   or README), Q1 is far off: we detect 56 chips, AMW 27; 30 chips are ours only, and only 8% of
   shared chips have identical scores (gaps up to 0.27, all six votes differ by up to 0.87).
   The comparison thread's 12:43 answer ("Q1 and Q2 match exactly on the same setup") was
   written 33 s after that run finished and relied on test_region alone.
2. **The deviation is systematic, not random: we detect more.** Pará Q1 is 30 vs 1 in our
   direction. It is the rainy season (median 4.6 clear observations per pixel vs 16.6 in Q2), so
   one extra or missing scene changes the median a lot. The differences are spread over all
   13 tiles that have shared chips, which points to a scene-level
   difference, not a code bug. Code is the same as AMW's: Q2 matches to 1e-6 in both regions.
3. **The 'last day' is a confirmed design difference, not just a hypothesis.** AMW passes the
   quarter's last day as Earth Engine's (exclusive) end date (`core/periods.py` `date_span`, gee.py
   filterDate), so 31 Mar / 30 Jun / 30 Sep are dropped; our `_status.json` shows end
   `2026-04-01` etc. So our quarters contain one more day of imagery by construction. This is a
   real deviation that the comparison accepts rather than removes.
4. **Choosing 2026Q2 for the 3-day subregion 3 run "because it matched exactly" selects on the
   outcome.** Q2 is the quarter where the window difference happened not to matter in both test
   regions; it can't reveal the Q1/Q3-type gap. Q1 is the informative quarter.
5. **test_region's Q1–Q3 were scored from the image cache** (`image_cache` set in each
   `_status.json`); Pará was not. Not cheating (the cache holds our own earlier downloads, keyed by
   our window `..._2026-04-01`), but the "exact" result was never shown on freshly fetched imagery
   until Pará Q2 did it.
6. **Things I checked that hold up:** the key join is correct (geometry match gives the same
   pairs, no duplicates; the only unmatched AMW rows inside our area are neighbouring tiles' chips
   overlapping our edge by about half); thresholds are strict `> 0.4` on both sides; the
   test_region numbers in the README reproduce exactly; the 4 test_region Q3 disagreements do
   sit between 0.35 and 0.45. One small gap: the notebook says `should have matched` "must be 0"
   but never asserts it (compare.ipynb cell 13).

## What would settle it

- Run `scene_dates.py` for Pará Q1 (`--run tests_output/test_region_para_2026-10-19 --quarter
  2026Q1`): if 31 Mar has clear scenes over the area, the last day explains it.
- Cleaner: re-score Pará Q1 with AMW's exact window (end 2026-03-31). If it then matches to 1e-6,
  the only cause is the window; if not, it is late ingestion / Cloud Score+ timing.

## Result of the rerun (13:35 UTC)

Ruby ran AMW's own `core/inference_pipeline.py` on the Pará test area for 2026Q1 twice,
ending 2026-03-31 (AMW's) and 2026-04-01 (ours). `amw_comparison/compare_rerun.py`:

| pair | A | B | both | only A | only B | identical | largest gap |
|---|---|---|---|---|---|---|---|
| published vs AMW code, end 03-31 | 27 | 27 | 27 | 0 | 0 | 100% | 0 |
| ours vs AMW code, end 04-01 | 56 | 56 | 56 | 0 | 0 | 100% | 0 |
| ours vs published | 56 | 27 | 26 | 30 | 1 | 8% | 0.27 |

The whole Pará Q1 gap is the last day (31 March). Our code reproduces AMW's code exactly;
the difference is only the end date.
