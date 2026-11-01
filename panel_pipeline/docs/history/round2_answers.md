> **History, not maintained.** 8 Oct. Still the deepest reasoning on overlap and SAM2. The 'every other chip is an exact 480 m grid' claim is corrected in `../open_design_decisions.Rmd` D6.

# Round 2: your decisions, my checks, and the SAM2 design

Everything here was checked in the code on the `code-walkthrough` branch (commit
`36718a8`), not in comments. Where a number comes from the authors' own design
notes rather than my own check, I say so.

---

## 1. Decisions so far

| Topic | Decision | Notes |
|---|---|---|
| Panel unit | Their overlapping 48 px chips, 24 px step | Reasons against are in §2. It can be undone later (see §2) |
| Deduplication | Keep their concat + dedupe, keeping the higher confidence | §3 explains what it actually fixes |
| Periods | 2018 to 2026, monthly and quarterly, no annual | I assumed 2026 ends at September / Q3. Please confirm |
| GEE calls | Separate call per month and per quarter (different medians) | Correct, see §5 |
| End dates | Their code stays as is; we pass the 1st of the next month as the end | Agreed. It also gives us a free consistency check (§5) |
| Cloud recipe | Same for monthly and quarterly; clear-image counts as quality variables | |
| Region | All of Amazon ACA, Andes handled with the same logic | |
| Andes | No second inference run; a threshold rule on the same chips | §4 shows this is safe |
| Ensemble | Keep all 6 member votes for every chip | §6 |
| postprocess | Kept as variables, not as a filter | |
| persist-detections | Kept as variables, not as a filter (I read "like 3" as "like postprocess") | Needs redesign because there are no annual periods (§8) |
| SAM2 | Masks only the period's own detections; filling in from earlier periods happens later, in the panel | My reading of "backfill before". Please confirm. Design in §7 |
| Base branch | You named **Pipeline Rebuild** | That branch does not exist on GitHub yet. Only `main` and `code-walkthrough` exist |

---

## 2. Overlapping chips: the case against (you asked for strong reasons)

How the overlap arises: `cut_chips` (`core/tile_utils.py:231`) steps through
the 576 px tile 24 px at a time with a 48 px window. That gives 23 × 23 = 529
chips per tile, and every pixel sits in up to 4 chips.

Reasons it might not make sense as an econometric unit:

1. **The observations are not independent.** One mine scar lights up a block of
   overlapping chips. A regression treats them as 4 separate findings when they
   are one. Your standard errors will be too small unless you cluster them, for
   example by tile or with Conley spatial standard errors. The effective sample
   size is roughly a quarter of the row count.
2. **Built-in spatial correlation.** Neighbouring chips share pixels, so their
   outcomes *and* their quality variables are mechanically correlated. Any
   spatial lag or spillover estimate picks this up as if it were a real effect.
3. **It bends the postprocess variable.** The isolation rule asks whether a
   chip's 5th-nearest detected neighbour lies within 3 km. A single scar already
   fires several overlapping chips, so a chip can "have 5 neighbours" just from
   its own scar. The rule mostly measures blob size, not clustering of mines.
4. **Areas add up wrong.** If you sum SAM2 area over chips, every pixel is
   counted up to 4 times. Totals must come from a non-overlapping partition.
5. **Size.** The panel is about 4 times larger than a non-overlapping one (§9).

Reasons to keep it, which I think win for the *dataset*:

- It is how the detector was designed and validated, so every chip is a case
  the model was built to score.
- It keeps every score, so nothing is lost.
- **It can be undone.** Taking every other chip in both directions gives an
  exact non-overlapping 480 m grid. One detail: each tile row holds 23 chips, an
  odd number, so "every other chip" must be counted globally (tile index × 23 +
  chip index), not inside each tile. Otherwise the pattern flips at every tile
  edge.

So: keep the overlap in the data, and decide in the analysis which version each
regression uses. I agree with you.

---

## 3. What concat really fixes (it is not the chip overlap)

The overlap *inside* a tile is deliberate, and dedupe leaves it alone:
neighbouring chips' centres are 240 m apart, and dedupe only removes chips whose
centres match to about 1 m.

What concat fixes is the **6 subregions**:

- They run the Amazon as 6 separate jobs, one per subregion file
  (`scripts/pipeline.py:151`, `Amazon_ACA_{1..6}.geojson`).
- Each job covers its own polygon with tiles. A tile that straddles the border
  between two subregions belongs to *both* polygons, so both jobs create it.
  Tiles sit on one fixed global grid, so it is the same tile with the same key
  and the same 529 chips.
- Those chips therefore appear twice: once in each subregion's file. Concat
  stacks the 6 files, and `dedupe_detections` keeps one copy per location, the
  one with the higher confidence.

I checked the subregion files (restored from the commit before they were
archived): the 6 subregions add up to the whole Amazon ACA (8.24 million km²),
with zero area overlap. So every duplicate comes from tiles on a border, never
from overlapping polygons.

**Why do the two copies differ at all?** The authors' notes say about half the
pairs differ by a median of 0.0005 in confidence. I believe I found the cause in
the code, though this part is my inference and not yet tested: fresh pixels from
Earth Engine are kept as decimals (`core/gee.py:365`), but cached pixels are
saved as whole numbers (`save_tile`, `astype` at `core/gee.py:501`, cuts off
decimals). A median of an even number of images can produce .5 values. So the
same tile scores slightly differently depending on whether it came from Earth
Engine or from the cache. **I propose rounding pixels to whole numbers every
time, before scoring.** Then a tile always gets the same score, and we can test
this on a pilot.

**Keeping it, with two small changes I'd suggest:**

- Keep the 6 subregions as our job split (they are natural units of work), and
  keep concat plus their max-confidence rule.
- Match duplicates on the exact chip ID (tile key + row + column) instead of a
  rounded centre point. For border duplicates the result is identical, because
  it is the same tile. But it removes a rounding risk the authors themselves hit
  (their notes describe 2024 matches collapsing at 6 decimals). Also record how
  many rows were dropped, as a check.
- When the two copies differ, we keep the row with the higher *mean* confidence,
  including all six of its votes, so votes and mean always come from the same
  scoring.

One change of scale: today only positive chips are written, so dedupe handles a
few duplicates. With every chip written, each border tile contributes all 529
chips twice. That is still a small share of the Amazon, but it is no longer
trivial.

---

## 4. Andes: is the threshold really the only difference? Yes, in inference

**Inference.** The two commands the driver writes (`scripts/pipeline.py:151-172`)
are identical except for `--region_path` (subregion vs `andes_supplemental`)
and `--pred_threshold` (0.4 vs 0.2). They use the same model, dates, image cache
and `--tries 3`, and everything else is a default.

**Same chips.** I measured the boundary files: the Andes area is 9,253 km², and
**100% of it lies inside Amazon ACA**. Tiles sit on one global grid, so the Andes
run creates tiles the Amazon run already created, and scores the same chips again.
The only difference in output is which chips pass the threshold (plus the tiny
cache rounding effect from §3; the authors measured a median gap of 0.0009).

So one inference run, storing every chip, gives you both. We then mark
`in_andes` = the chip touches the Andes boundary, which is exactly what
`geo_filter.py` does (`gdf.intersects(boundary)`).

**But downstream, Andes is treated differently in two more ways**, and "the same
logic" means copying these too:

1. **No isolation rule for Andes.** The Amazon detections go through the
   0.43/0.75 dual threshold. The Andes ones (≥ 0.2) do not, and they are added on
   top in `persistence.py` (`load_period` combines the two and keeps the higher
   confidence per location). So an Andes chip counts as detected at ≥ 0.2, with
   no isolation check.
2. **A separate SAM2 run.** `mask-annual` runs SAM2 twice: once on the Amazon
   loose set and once on the raw Andes set (`scripts/pipeline.py:175-190`). The
   masks are combined afterwards. Why this matters is in §7.

---

## 5. Monthly vs quarterly Earth Engine calls

You're right: a median over 3 months is not the median of three monthly medians.
So the quarter needs its own call, and each month needs its own call.

The clear-image counts (`n_clear`, `n_total`) are different: they are plain
counts. So **a quarter's count equals the sum of its three months' counts**,
provided the date windows meet exactly. With your end-date rule (end = 1st of
the next month, which Earth Engine excludes) they do. If any pixel fails
`n_clear(Q) = n_clear(M1) + n_clear(M2) + n_clear(M3)`, something is wrong with
dates or data. That is a free automatic check on every tile.

The counts go into the **same** call as the image, as extra bands. That removes
the second Earth Engine request the timeseries folder makes
(`timeseries/coverage.py:155`).

---

## 6. Ensemble votes

The `.h5` file has 6 member CNNs (`v4.10-CNN800kb`, `v4.18-…kd`, `v4.20-…kg`,
`v4.21-…ka`, `v4.22-…kb`, `v4.22-…kc`). Each ends in a single sigmoid output,
and their outputs are joined side by side. So `model.predict` already returns 6
probabilities per chip, and the "confidence" is their plain mean
(`_probs_from_predictions`, `core/inference_engine.py:786`).

Their code already keeps the votes, but only for positive chips and as a text
string (`inference_engine.py:830`). We keep them as 6 number columns for every
chip. That costs no extra computation; it only costs storage (§9).

---

## 7. SAM2, re-checked properly

### What their SAM2 actually does, step by step

1. **Prompt set.** A detections file: Amazon loose set, or raw Andes set.
2. **Where to look.** `_simplify_for_tiling` (`inference_engine.py:1555`) blurs
   the detections into blobs (buffer of 0.005°, about 550 m, out and back in,
   holes removed). Then **256 px tiles** (+12 px padding) are laid over the
   blobs (`sam2_mask.py:184`, so not the detector's 552 px tiles).
3. **Imagery.** Each 256 px tile fetches its own composite from Earth Engine
   (same recipe, same dates).
4. **The prompt.** `polygon_gdf_to_pixel_bbox` (`inference_engine.py:1342`)
   makes **one single box per tile: the bounding box around *all* detected chips
   in that tile**, not one box per chip.
5. **SAM2.** It runs on an RGB picture (bands 4, 3, 2 divided by 3000) with that
   box. It returns 3 candidate masks, and the one with the best score is kept.
6. **Spatial prior.** A penalty is added that is 0 inside detected chips and
   grows with the square of the distance outside them (σ = 12 px). The result is
   smoothed and cut at 0, giving the mask (`-msk.tif`). The smoothed scores are
   also saved, capped at ±16 (`-logits.tif`).
7. **No restart and no failure record.** It doesn't skip finished tiles, and
   failed tiles are only printed (`inference_engine.py:1655`).

### Correction to what I said last time

I suggested running SAM2 on the detector's 552 px tiles so the imagery
could be reused. **That was wrong, so I'm withdrawing it.** Because the prompt is
one box around everything detected in the tile, tile size changes the answer.
On a 552 px tile the box can cover several separate mines and most of the tile,
so the result would be a different, untested setting. The 256 px tiles are part
of how their masks are made.

And the extra download is small. Their notes put a basin-wide SAM2 mask at roughly 15,000 tiles
of 280 px for one annual period (`docs/design/persistence-planning.md:118`), against about 270,000 detector tiles of 576 px (my
estimate for Amazon ACA, see §9). That is about **1–2% of the detector's
pixels**. Keeping their design costs little and keeps their results.

### My proposal

Per period (each month and each quarter separately), after that period's
detections are finished:

| Step | What | Same as theirs? |
|---|---|---|
| a. Prompt sets | (1) Amazon: chips passing the loose dual threshold (≥ 0.4 raw, then 0.43/0.75). (2) Andes: `in_andes` chips with mean ≥ 0.2 | Yes, the same as their annual masks |
| b. Tiles and prompts | `_simplify_for_tiling`, 256 px tiles, one box per tile, prior, smoothing: unchanged | Yes |
| c. Two runs | Amazon set and Andes set masked separately, then combined pixel by pixel (mine if either says mine) | Yes, the same as their two runs and merged mosaic |
| d. Imagery | Fetch only bands 4, 3, 2 plus `n_clear` for SAM2 tiles | Same pixel values (SAM2 only reads those 3 bands), about 4× smaller download and cache |
| e. Restart | Skip tiles whose mask and logits exist; write a failed-tiles list; retry | New |
| f. Per-chip numbers | For every chip inside a SAM2 tile, detected or not: masked pixels, observed pixels (`n_clear > 0`), mask area in m², mean and max logit, and whether any SAM2 tile covered it | New: this is what goes into the panel |

Two points that matter for the econometrics:

- **Why every chip in a SAM2 tile, not just detected ones:** their notes say
  SAM2 often segments scars that extend past the chip that prompted them
  (`persistence-planning.md:99`). Counting mask pixels in every chip's
  footprint captures that extent in the neighbouring chips.
- **SAM2 area is only measured where detection fired.** A chip outside every
  SAM2 tile gets area 0 by construction, not because it was measured as 0. So the
  panel needs a `sam2_covered` flag, and area should be read together with it.
  This is a selection issue your analysis must handle.
- **Monthly cloud gaps.** Pixels with no clear image come back as 0 in all bands
  (black). SAM2 sees them as black holes. So area must be reported over
  *observed* pixels, and the observed share is itself a quality variable.

**Combining overlapping SAM2 tiles.** Neighbouring SAM2 tiles overlap in their
12 px padding. Their mosaic step combines overlaps as "mine if any tile says
mine" (`sam2_build_cog.py`, `build_mask_union_vrt`). We use the same rule when
counting pixels per chip.

**Counting pixels per chip cheaply.** Chips and mask pixels are both upright
rectangles in latitude/longitude. So for each SAM2 tile we can build a "running
total" table of the mask (each cell holds the sum of everything above and to the
left of it). Then the pixel count inside any chip takes 4 lookups, whatever the
chip's size. This is called a *summed-area table*. I'll explain it properly when
we write that file.

### Order, parallel runs, storage

```
for each period P (periods are independent, so many run side by side):
  detect P, subregion 1..6        (CPU jobs, Earth Engine limited)
  -> concat + dedupe P            (one small job)
  -> postprocess variables P      (needs the whole period: neighbours cross subregion borders)
  -> SAM2 P, Amazon + Andes       (GPU jobs, shards of SAM2 tiles)
  -> per-chip SAM2 numbers P
after all periods:
  persistence variables           (needs every period)
  -> final panel
```

- **Why postprocess must wait for all 6 subregions:** the 3 km neighbour search
  crosses subregion borders. Running it per subregion would make border chips
  look isolated.
- **Earth Engine is the bottleneck, not the cluster.** Earth Engine limits how
  many requests your project can have running at once, however many cluster nodes
  you use. So the "GEE queue" is a cap on how many fetching jobs run at once. In
  Slurm (the common cluster scheduler) this is one setting on a job array, e.g.
  `--array=0-999%8` means "1000 jobs, at most 8 at a time".
- **Storage of SAM2 outputs** (estimate, using their annual tile count; months may differ): about 15,000 tiles per period ×
  (mask about 5 KB + logits about 34 KB; the logits size is from their notes) ≈
  0.6 GB per period, or about 80 GB for 140 periods. The logits let you re-run
  the mask at another cut-off later (a robustness check). Without them it is a
  few GB.
- **SAM2 imagery cache:** with only 3 bands plus the count, about 9 GB per
  period. Delete it once that period's per-chip numbers have been checked.
- **Detector imagery cache:** see §9. It cannot be kept.

---

## 8. Persistence as variables (no annual periods)

Their rule (`compute_onsets`, `core/persistence.py`) only accepts **annual**
periods as possible start dates. It needs a detection in year Y and again within
2 years. With only months and quarters it cannot run unchanged. My proposal, all
computed from the finished panel:

Looking back (safe to use as controls):
- `det_prev`: detected in the previous period of the same frequency
- `n_det_before`: number of earlier periods with a detection
- `first_det`: first period with a detection; `periods_since_first`

Looking forward (only for outcome definitions and robustness, clearly marked):
- `confirmed_k_n`: detected at least k times within the next n periods. This is
  their idea, but with the window measured in calendar months so monthly and
  quarterly match
- `onset`: first period whose own forward window passes `confirmed_k_n`, which is
  their onset definition adapted

Across frequencies:
- `det_in_quarter`: for a month, whether the quarter containing it was
  detected. This is their "quarterly witnesses" option turned into a variable

Their settings were k=2 within 2 years. **You need to choose k and n for months
and quarters.** I suggest k=2 within 12 months as a starting point.

---

## 9. Scale (what this whole thing means in numbers)

These are estimates from the boundary file and the code, not measured runs:

- Amazon ACA: 8.24 million km² ÷ 30.5 km² per tile ≈ **270,000 tiles** (the
  real count is a little higher because of edge tiles). Their notes list 219,800
  for the older `amazon_basin` boundary.
- 529 chips per tile ≈ **143 million chips**.
- Periods: 105 months (Jan 2018 to Sep 2026) + 35 quarters = **140**.
- **About 20 billion chip-period rows**, and about 38 million Earth Engine
  requests for detection.
- Panel storage: 6 votes as 2-byte numbers plus about 6 bytes of quality
  variables ≈ 18 bytes per row ≈ **360 GB before compression**. We decide the
  number format when we write that file.
- Detector imagery: about 8.6 MB per tile per period uncompressed → roughly
  **200 TB** if it were all kept. So detector imagery cannot be cached long-term.
  It stays in memory (or on the node's local disk for the length of one job), and
  restarts work from finished-tile files instead (see §10).

**Earth Engine cost is the open risk.** I can't see your Earth Engine project's
limits. Non-commercial projects have caps on simultaneous requests and on monthly
compute, and 38 million requests may exceed them. The first thing to run is a
pilot that measures time and Earth Engine compute per tile, then we scale the
plan.

---

## 10. The two items you asked about

### Producer/consumer (`bulk_predict`, `core/inference_engine.py:1115`)

Picture a kitchen. Several **waiters** (producer threads) each go to the market
for one tile's ingredients: they ask Earth Engine for the pixels, or read them
from the cache. One **chef** (the consumer thread) cooks: cuts chips, runs the
CNN, keeps the positives. Between them is a **counter with 64 places** (the
queue, `queue.Queue(maxsize=64)`, line 1167).

- *Why many waiters:* fetching is mostly waiting for the network. A thread that
  waits costs almost nothing, so 8 (`max_workers`) can wait at once
  (`ThreadPoolExecutor`, line 1189).
- *Why one chef:* there is one model object in memory. Calling it from several
  threads at once is not safe, and a GPU works best on one stream anyway. A lock
  (`_tf_model_lock`, line 430, used at 765) makes sure only one thread uses it
  at a time.
- *Why the counter has a limit:* if waiters are faster than the chef, the counter
  fills up. At 64 waiting tiles, waiters must stop. This keeps memory bounded and
  is called *backpressure*.
- *How the chef knows to stop:* when all waiters of a batch are done, a special
  object, the **sentinel**, is put on the counter (line 1210). The chef stops
  when it picks it up.
- *Batches of 500 tiles:* the waiter team is created per batch of 500 tiles
  (`max_concurrent_tiles`). Everyone must finish before the next batch starts,
  so one slow tile holds up the whole batch. This is called the *straggler*
  problem. We'll keep the structure but improve that.

### The batch rewrite and the retries (lines 1237 and 1242)

After each batch, the new positives are added to everything found so far, and
**the whole output file is written again from scratch** (`predictions.to_file`,
line 1237). After all batches, tiles that failed are tried again, up to
`--tries` times (`retry_tiles = fails`, line 1242).

Problems with this, for us:

1. **It gets slower as it goes.** Batch 1 writes 1 batch of results, batch 100
   writes 100 batches' worth. Total writing grows with the *square* of the number
   of batches. For positives only that's tolerable. With every chip written, the
   file would hold millions of rows rewritten 540 times per period.
2. **A crash during writing can break the file.** It writes directly over the
   old file, so a crash mid-write leaves half a file.
3. **Restarting throws the results away.** The function starts with an empty
   table (line 1142) and its first write replaces the old file. A crash means
   re-scoring everything; only the image cache makes it faster.
4. **Failed tiles disappear.** After the last try, only the *number* of failed
   tiles is logged (line 1241), not which ones. In the output, "no mine" and
   "never scored" look the same.

What we'll do instead: one small result file per tile, written under a temporary
name and then renamed. A rename is all-or-nothing on Linux file systems, so a
file either exists complete or not at all. The set of finished files *is* the
"done" list, so a restart skips them. Failed tiles go into their own file, and
the panel marks them as missing.

---

## 11. Proposed order for writing the code together, file by file

1. `config.py` + `periods.py`: settings; months and quarters; dates
2. `tiles.py`: tile plan per subregion; chip ID; `in_andes`
3. `fetch.py`: Earth Engine composite + count bands (adapted `gee.py`)
4. `detect.py`: adapted engine; every chip, 6 votes, per-tile files, done and failed lists
5. `concat.py`: concat + dedupe per period
6. `postprocess_vars.py`: dual-threshold variables
7. `sam2_run.py`: SAM2 per period, Amazon + Andes
8. `sam2_chips.py`: per-chip SAM2 numbers
9. `persistence_vars.py`
10. `panel.py`: final table
11. `jobs/`: Slurm scripts + a driver like `pipeline.py`

## 12. Open questions

1. Does 2026 end at September / Q3?
2. Should **Pipeline Rebuild** be created from `main` or from `code-walkthrough`?
   Only `code-walkthrough` has the driver, `periods.py` and `persistence.py`.
3. Persistence: which k and n (my suggestion: 2 within 12 months)?
4. Is my reading of SAM2 "backfill before" right (§1)?
5. Cluster: is it Slurm? Are GPUs available, and is there a storage quota?
6. Which Earth Engine project will this run under, and what are its limits?
