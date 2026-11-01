> **History, not maintained.** a results record (19 Oct), summarised in `../findings.md` §1. Kept in full for the per-month tables.

# Reading the notebook: test_region, 2026 (monthly Jan–Sep, quarterly Q1–Q3)

Run `tests_output/test_region_2026-10-19`, subregion 2: 36 tiles, 19,044 chips, scored on 19 Oct
with real Earth Engine images. All numbers below were recomputed from the run's part files.

## 1. Did the run work? Yes, completely

- All 12 tasks are done: 36/36 tiles each, no failed tiles, one session each.
- Every integrity check passes: every planned chip is present exactly once, nothing is
  unplanned, confidence is the mean of the 6 votes, n_clear ≤ n_obs, and no values are missing.
- Time: about 2 minutes per period (1.7–5.2 min) with 6 download workers. That is about
  3.3 s per tile on a laptop.
- Image cache: 432 images (36 tiles × 12 periods), 1.7 GB. That is about 4 MB per image, half
  my earlier estimate.
- `dirty_code = True` only means config.yaml and the notebook had uncommitted edits while it
  ran. No scoring code was affected, but commit before a run so `git_commit` describes it fully.

## 2. The three image-count variables (the confusing part)

All three are computed **per pixel** and then **averaged over the chip's 48 × 48 = 2,304
pixels**. That is why they have decimals.

| variable | per pixel it counts | read it as |
|---|---|---|
| `n_obs` | Sentinel-2 images of that pixel in the period that have a cloud score (cloudy or clear) | how often the satellites passed over |
| `n_clear` | the ones that pass AMW's cloud test (`cs_cdf ≥ 0.6`) | how many images the median composite, which is what the model sees, is made of |
| `empty_share` | (share of pixels where `n_clear` = 0) | the part of the chip that is blank in the composite: the model sees no data there |

Example: `n_obs = 7.0, n_clear = 1.30, empty_share = 0.11` means the satellites passed 7 times
that month. On average 1.3 of those passes were clear for a pixel, and 11% of the chip's pixels
had no clear pass at all.

**Why n_obs is almost constant (7.0, 7.09, …):** it depends on the satellites' fixed revisit
schedule, not on the weather. Every chip in a month gets the same number, except where two
orbit tracks overlap. Those pixels are imaged twice as often, which is why the maximum is about
2 × the minimum (7 vs 16 in January).

**n_clear is the one that varies:** clouds.

## 3. Image counts per month

| period | n_obs | n_clear (mean) | 10% of chips have fewer than | empty_share (mean) | chips with n_clear < 1 | chips > 0.4 |
|---|---|---|---|---|---|---|
| 2026-01 | 7.1 | 3.0 | 1.2 | 4.6% | 7.2% | 88 |
| 2026-02 | 6.1 | 5.0 | 4.0 | 0.1% | 0.1% | 71 |
| 2026-03 | 7.1 | 2.7 | 2.0 | 0.9% | 1.6% | 19 |
| 2026-04 | 8.1 | 3.3 | 1.7 | 2.1% | 3.4% | 41 |
| **2026-05** | 9.1 | **1.2** | **0.1** | **21.7%** | **37.0%** | 63 |
| 2026-06 | 9.1 | 2.8 | 1.3 | 2.6% | 4.5% | 109 |
| 2026-07 | 10.1 | 3.7 | 1.9 | 2.2% | 3.2% | 106 |
| 2026-08 | 9.1 | 4.4 | 2.5 | 0.6% | 0.9% | 143 |
| 2026-09 | 8.1 | 5.8 | 3.8 | 0.1% | 0.1% | 138 |
| 2026Q1 | 20.3 | 10.8 | 8.0 | 0.0% | 0.0% | 84 |
| 2026Q2 | 26.3 | 7.3 | 4.1 | 0.4% | 0.6% | 105 |
| 2026Q3 | 27.3 | 13.9 | 9.2 | 0.0% | 0.0% | 146 |

How to read it:
- **May is the cloudy month.** A typical pixel had about one clear image, and 37% of chips had
  less than one on average. About a fifth of all pixels were blank. The other months have about
  3–6 clear images.
- **No chip with n_clear < 1 is detected**, in any month or quarter. Missing data does not make
  false detections here. It can hide real ones, though, so May's count (63) is probably lower
  than what is on the ground.
- **Detections rise from March (19) to August (143).** This partly follows n_clear (August and
  September are clearer) but not fully: March has 2.7 clear images and only 19 detections. From
  these numbers alone we can't separate real change from image quality. That is exactly why
  n_clear and empty_share are stored next to every score: later analysis can control for them,
  or drop chip-periods with too little data.

## 4. Monthly vs quarterly

- **The windows line up exactly.** For all 57,132 chip-quarters, the quarter's n_obs and n_clear
  equal the sum of its three months, with a maximum difference of 0 (right-hand scatter: every
  point on the line). Months and quarters cover the same days with no gap or overlap, so the
  two panels are compatible.
- **A quarter is its own image, not an average of three months.** Its composite is the median
  of 7–14 clear images instead of 1–6, with almost no blank pixels. So quarterly inputs are
  cleaner.
- **Quarterly scores are usually at or above the mean of the three monthly scores** (left
  scatter: most points sit above the line). With a cleaner composite the model is more
  confident on real features.
- **Agreement on detections (> 0.4):**

  | | detected in at least one month | in no month |
  |---|---|---|
  | **detected in the quarter** | 316 | 19 |
  | **not detected in the quarter** | 74 | 56,723 |

  The quarter catches most of what any month catches (316 of 390). The 74 detected only in a
  month are mostly single-month flickers. The 19 detected only in the quarter are where the
  cleaner composite pushed a borderline chip over 0.4.
- **Map, January vs Q1:** the same features in the same places (strings along the rivers), with
  88 and 84 chips above 0.4.

## 5. What the full run would weigh (section 9)

- Scores take 45.6 bytes per row on disk, so the whole Amazon would be about 690 GB monthly and
  230 GB quarterly.
- Speed: one laptop at this rate (3.3 s per tile) would need about 10 days per period for all
  ~275,000 tiles. That is about 4 years for the full 140 periods, so the cluster's parallel
  tasks are what make the full run possible. With many tasks, the limit becomes Earth Engine's
  request quota rather than our machines.
