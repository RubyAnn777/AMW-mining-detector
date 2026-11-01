> **History, not maintained.** first review of AMW's 13 stages (8 Oct). Its verdicts to drop concat, postprocess and persistence were overruled by Ruby: they are kept as variables (see `../STATE.md`).

# Which of the existing pipeline stages do we need?

Reviewed from the code itself (comments ignored), on the `code-walkthrough` branch
of RubyAnn777/AMW-mining-detector at commit `36718a8`. That branch is the only place
the stage driver exists: `scripts/pipeline.py`, `core/pipeline_config.py`,
`core/periods.py`, `core/persistence.py` and `scripts/stage_outputs.py` are **not on
`main`**. `main` only has the underlying scripts (inference, concatenate, geo_filter,
postprocess, SAM2).

---

## Summary

| # | Stage | What it really does | Verdict for us |
|---|-------|--------------------|----------------|
| 0 | review-config | Prints the settings in `pipeline_config.py`. Runs nothing. | **Adapt** (replace with a real config file + validation) |
| 1 | inference | Prints `inference_pipeline.py` commands; a human runs them | **Keep the engine, adapt heavily** |
| 2 | concat | Joins the 6 subregion files and removes duplicate chips on the seams | **Drop** (design it away) |
| 3 | filter | Clips Andes low-threshold detections to the Andes boundary | **Drop** (unless you want the Andes pass, see Q5) |
| 4 | postprocess | Dual threshold + "isolated chips need higher confidence" | **Drop from pipeline, keep as an analysis variable** |
| 5 | persist-detections | Decides a mine "really" started only if seen again the next year | **Drop** (it looks into the future; bad for a panel) |
| 6 | mask-annual | SAM2 on each year's detections | **Adapt** (one SAM2 rule for every period) |
| 7 | mask-quarterly | SAM2 only on the *new growth* in a quarter | **Drop as designed**, replaced by 6 |
| 8 | cog | Mosaics SAM2 tile masks into map rasters (COGs) | **Drop** (we aggregate to chips instead) |
| 9 | persist-masks | Builds "onset year" rasters from the masks | **Drop** |
| 10 | stage | Copies products into folders laid out for upload | **Drop** |
| 11 | manifest | Prints "edit MANIFEST.yaml by hand". Runs nothing. | **Replace** with an automatic run log |
| 12 | publish | Prints `gsutil`/`aws` upload commands | **Drop** |

So, as you guessed, we only go two stages deep (inference, then SAM2), and then add
stages the old pipeline never had: chip panel assembly, quality variables, and a
job manager for the cluster.

---

## How the driver and the CLI work (the computer science part)

### What happens when you type a command

`python scripts/pipeline.py concat filter --periods Q125 Q225`

1. The shell splits that line into words and hands Python the list
   `["scripts/pipeline.py", "concat", "filter", "--periods", "Q125", "Q225"]`
   (called `sys.argv`, "argument vector").
2. `main()` (`scripts/pipeline.py:558`) builds an **argparse** parser. argparse is the
   standard Python library that turns that word list into named values. Each
   `add_argument` line is a rule:
   - `"stages", nargs="*"` (line 561): a *positional* argument. Every bare word goes
     into a list called `stages`, so `["concat", "filter"]`.
   - `"--periods", nargs="+"` (line 562): an *option*. Everything after `--periods`
     until the next `--something` goes into `periods`, so `["Q125", "Q225"]`.
   - `"--all", action="store_true"` (line 567): a *flag*. It is `True` if present,
     `False` if not. Same for `--dry-run` and `--list`.
3. `ap.parse_args()` returns one object, `args`, with `args.stages`,
   `args.periods`, `args.use_all`, and so on.

**So the answer to "is `--periods` defined somewhere?"**: yes, at
`scripts/pipeline.py:562`. Nothing outside that file knows about it.

### How it decides which periods a stage works on

`scripts/pipeline.py:597-632`:

- With `--all`: every period in `ALL_CURRENT_PERIODS` (`core/pipeline_config.py:53`).
- With `--periods X Y`: only X and Y, but **each must already be in
  `ALL_CURRENT_PERIODS`**, or it stops with an error.
- Some stages (`WHOLE_HISTORY`, line 69: persist-detections, persist-masks, stage,
  manifest) *ignore* `--periods` and always use the whole list, because they
  recompute the full history.

So to your question "do I add periods to the list or just pass them on the CLI?":
**in their design, both** — the list is the master copy, the CLI picks a subset.
That double bookkeeping exists because they add one quarter at a time. We score
everything in one go, so for us the list *is* the run, and the CLI option is
only useful for "re-run just these periods".

### "HUMAN" stages vs "pipeline" stages

`HUMAN` (line 62) = review-config, inference, mask-annual, mask-quarterly,
publish. For these the driver does **not run anything**: it *prints* the commands
(`EMITTERS`, line 637) and a person copies them into terminals (`tmux`) on the VM.
The other stages (`RUNNERS`, line 641) are run by the driver itself through
`subprocess.run` (line 312), which means "start another Python program and wait
for it to finish".

For us this is the key thing to change. On a cluster, a person pasting commands
does not scale. The cluster's **job scheduler** (usually Slurm: you submit a
"job script", it queues it and runs it when a GPU/CPU node is free) takes that
role. The "print the commands" idea is still useful though: our driver can
*generate* job scripts the same way, from one config, so nobody types dates.

### How stages find each other's files: "the filename is the contract"

No stage passes data to the next in memory. Each stage writes files, and the next
stage *rebuilds the expected filename* from the same settings and looks for it,
for example `Amazon_ACA_{MODEL}_{RAW_TAG}_{start}_{end}.geojson`
(`scripts/pipeline.py:338`). That is why the thresholds are in
`pipeline_config.py`: they appear in filenames, and if one stage disagrees it
silently finds nothing.

- **For keeping it:** simple, inspectable, every stage restartable on its own.
- **Against:** fragile. A mismatch means "no file found", not an error: filter and
  postprocess print a message and carry on with exit code 0
  (`scripts/pipeline.py:393` and `:417`). Behaviour parameters such as
  `clear_threshold` are *not* in the filename, so outputs made with different
  cloud settings get the same name.
- **What I'd do:** keep "files between stages", but make each output carry its own
  settings inside it (columns or file metadata) and have each stage check them,
  rather than trusting the name.

### "Idempotent" stages

Almost every runner starts with `if dest.is_file(): continue` — if the output
exists, skip. That is called *idempotent*: running twice gives the same result as
once. It is how they "restart" after a crash. This is a concept we definitely
keep, but at a finer level (see stage 1).

---

## Stage by stage

### 0. review-config — Adapt

**What the code does:** `cmds_review_config` (`scripts/pipeline.py:108`) only
prints the model name, periods, subregions and thresholds from
`pipeline_config.py`. It changes and checks nothing.

**Keep:** the *idea* that every shared setting lives in one place, and that the
period list is derived into dates by one function (`Period.parse(...).date_span`,
`core/periods.py`). Their dates are never typed by hand, which is good.

**Change:**
- `Period` only understands years (`2025`) and quarters (`Q125`). There is no
  month. We need months, and a Period type that can tell you which months make up
  each quarter, so the two frequencies line up.
- Replace "print and hope a human reads it" with a config file (e.g. YAML) that
  the pipeline *validates* and then *copies into the output folder* of every run,
  together with the git commit. That copy is your provenance: two years later you
  can say exactly what produced a number.
- Their list mixes annual periods in. Do you want annual too? (Question 2.)

**Objection to myself:** a Python file as config (like theirs) is fine and lets
you compute lists (e.g. "every month 2018–2025") instead of typing 96 entries.
Either is OK; the important part is validation + a frozen copy per run.

### 1. inference — Keep the engine, adapt heavily

**What the code does:** for each period and each of 6 subregions, the driver prints
a `core/inference_pipeline.py` command (`scripts/pipeline.py:151`). That script:

1. Cuts the region into tiles of 552 px (+12 px padding) on a fixed global UTM grid
   (`tile_utils.create_tiles`).
2. For each tile, asks Earth Engine for a cloud-masked **median composite** of all
   Sentinel-2 images in the period (`core/gee.py`, `_build_composite` and
   `get_tile_data`): each image's pixels with Cloud Score+ below 0.6 are hidden,
   then the median of what is left is taken.
3. Cuts the tile into 48 px chips with a 24 px step (so chips **overlap by half**),
   runs the CNN ensemble on them, and gets a probability per chip.
4. **Keeps only chips above the threshold** (`inference_engine.py:803`,
   `idx = np.where(mean_preds > pred_threshold)`). Everything else is thrown away.
5. Uses a *producer/consumer* setup (`bulk_predict`, line 1115): several threads
   download tiles from Earth Engine at once (producers, because downloading is slow
   and mostly waiting), one thread runs the model (consumer, because the model is
   one shared object). A queue sits between them.
6. After each batch of 500 tiles, rewrites the whole output GeoJSON
   (line 1237). Failed tiles are retried up to `--tries` times (line 1242).

**Keep:**
- The tiling and chip cutting. Tiles sit on a fixed global grid and chips are cut
  at fixed offsets inside them, so **the same chip has the same location in every
  period**. That is exactly the "aligned per chip" property you need for monthly vs
  quarterly comparison.
- The model, the normalisation (`pixels / 10000`, clipped), the composite recipe
  (unless we decide to change it for monthly, see Q3).
- The producer/consumer idea and the per-tile image cache.

**Change (strong reasons):**
- **Write every chip, not just positives.** For a panel, "probability 0.05" and
  "not observed because cloudy" are different values, and both are information.
  Dropping sub-threshold chips makes the panel unbalanced and confuses "no mine"
  with "no data". Store the probability for every chip, every period, and
  threshold later in the analysis.
- **Quality variables in the same Earth Engine call.** `core/gee.py` makes one
  request per tile (`computePixels`, line 336). The timeseries code makes a
  *second* full request per tile just to count clear images
  (`timeseries/coverage.py:155`). The count can be added as two extra bands
  (`n_clear`, `n_total`) to the same image before the single request, so one call
  returns both. Halves the Earth Engine traffic.
- **Restart properly.** Today the "progress" is one GeoJSON rewritten after each
  batch; there is no record of *which* tiles are done, so a crash means starting
  the region again (only cached images help). Tiles that fail all retries are just
  logged as a count and silently missing from the output. We want: one small
  output file per tile (or per batch) written as soon as it is done, a "done" list
  so a restart skips finished tiles, and an explicit "failed tiles" file that the
  panel records as missing rather than as zero.
- **Date bug:** Earth Engine's `filterDate(start, end)` excludes the end date, and
  the code passes the last day of the period (e.g. `2025-03-31`). So every period
  loses its last day. Annually that is small; for a month it is ~3% of the
  images. We pass the first day of the next period instead.
- **Subregions:** they run 6 subregions as separate jobs, which produces duplicate
  tiles on the seams (that is why "concat" exists). We split work by *tile list*
  instead, so every tile is scored exactly once.
- **Cluster:** the GPU model and the Earth Engine downloads have different
  bottlenecks. On a cluster it is usually better to download and cache imagery in
  one kind of job (CPU, many parallel requests, limited by Earth Engine quota) and
  score in another (GPU, reads only from cache). That is the "GEE queue" you
  mentioned.

**Against changing too much:** the engine is the most tested part of the code and
the model was trained on exactly this composite recipe. Any change to how
pixels are made (cloud threshold, median, end date) changes what the model sees.
I'd keep the image recipe identical except the end-date fix, and validate on a
period where we have their published outputs.

### 2. concat — Drop

**What the code does:** joins the 6 subregion files for a period and removes chips
that appear twice on subregion seams, keeping the higher confidence
(`scripts/concatenate.py`, `postprocess.dedupe_detections`). Dedup works by
rounding chip centres to 5 decimals.

**Why drop:** it only exists to repair a problem created by splitting the run into
overlapping subregions. If we build one deduplicated tile list up front, there is
nothing to repair. Our panel key will be the tile key + chip row/column, which is
exact, instead of a rounded coordinate.

**For keeping it:** none for us, beyond "it's already written".

### 3. filter — Drop (pending Q5)

**What the code does:** the Andes area is scored a second time with a lower
threshold (0.2 instead of 0.4); `geo_filter.py` keeps the detections touching the
Andes boundary.

**Why drop:** this is a *threshold* trick. If we store every chip's probability,
there is no second pass: a different threshold for Andes chips is a line in the
analysis. It also removes a second full inference run over the Andes.

### 4. postprocess — Drop from pipeline, keep the logic for analysis

**What the code does:** keeps chips with confidence ≥ 0.43, then computes each
chip's distance to its 5th nearest detected neighbour; if that is more than 3 km
("isolated"), it needs ≥ 0.75 (`core/postprocess.py`, `dual_threshold_filter`).
Done twice, for "loose" (0.43/0.75) and "stringent" (0.55/0.8).

**Why not in the pipeline:** it is a modelling decision ("isolated detections are
probably false") applied to the *data*. In econometrics you want the raw measure
in the panel and these rules as variables or robustness checks. Also, the
neighbour count depends on how many other chips were detected that period, so a
cloudy month (fewer detections) makes chips look "isolated" and pushes them out:
that ties the outcome to cloud cover, which is exactly what your quality
variables are supposed to control for.

**Keep:** the code is clean and reusable. A "neighbours within 3 km above 0.43"
column could be computed later from the full panel.

### 5. persist-detections — Drop

**What the code does:** `core/persistence.py` (`compute_onsets`) says a location
"really" became a mine in year Y only if it is detected in Y *and* again within
the next year (k=2 of a 2-year window, annual periods only). Then it builds
cumulative layers and "growth" (new) files per quarter.

**Why drop (strongly):** it uses **future** observations to define the present.
Whether 2022 counts depends on 2023. In a panel this creates built-in correlation
between periods and a variable that cannot be used to study timing without
mechanically biasing results. Persistence rules, if you want them, belong in the
analysis where you can show results with and without.

**For keeping:** it is a sensible denoising rule for a public map, which is what it
was built for. Not our purpose.

### 6. mask-annual — Adapt; 7. mask-quarterly — Drop as designed

**What the code does:** `core/sam2_mask.py` takes a detections file and dates,
re-tiles the detections into 256 px tiles, gets imagery for each, and runs the
fine-tuned SAM2 model, which outputs a pixel mask (and logits) per tile
(`SAM2_Masker.bulk_mask_polygons`, `inference_engine.py:1595`). Annual: run on
loose postprocessed detections. Quarterly: run only on the *growth* file from
persist-detections, then OR-merged onto last year's mask.

**Keep:** the SAM2 model, `SAM2_Masker.predict` (prompting with the chip boxes,
spatial prior, smoothing).

**Change:**
- **One rule for all periods.** Annual and quarterly are prompted differently
  today, so their areas aren't comparable. For the panel, every period (month and
  quarter) should be masked the same way, from that period's own chip scores.
  Which chips to prompt is a decision for you (Q4).
- **Imagery downloaded twice.** SAM2 uses 256 px tiles (`sam2_mask.py:184`) and
  the cache filename contains the tile key, which contains the tile size
  (`tile_utils.py:79`, `gee.py:301`). So SAM2 can never reuse the imagery the
  detector already downloaded; it asks Earth Engine again. Running SAM2 on the
  same 552 px tiles (or cutting from the cached ones) removes that.
- **No restart, no failure record.** It does not skip existing masks, and a failed
  tile is only printed (`inference_engine.py:1655`). Same fix as for inference.
- **Output per chip.** Instead of map rasters, compute mask area (and mean logit)
  per chip per period and write that into the panel. The full masks can be kept
  or deleted (that's the caching question: imagery cache is the big storage cost;
  masks are small).

### 8. cog, 9. persist-masks, 10. stage, 11. manifest, 12. publish — Drop

- **cog** (`sam2_build_cog.py`) turns many tile masks into large map files
  (Cloud-Optimised GeoTIFFs) for web display, via GDAL. We aggregate to chips, so
  we don't need maps. Keep the option for a few visual checks.
- **persist-masks** (`sam2_persistence.py`) applies the same future-looking onset
  logic to masks. Same objection as stage 5.
- **stage** (`stage_outputs.py`) copies files into the folder layouts of the two
  public buckets. Website-only.
- **manifest** only prints a reminder to edit a YAML by hand (`pipeline.py:541`).
  Replace with an automatic run record (config + git commit + counts of done /
  failed tiles) written by the pipeline itself.
- **publish** prints upload commands for Google Cloud and Source Cooperative.
  Nothing to publish.

One idea worth stealing from publish: they check uploads **by listing file
names**, not by counting, because tools silently dropped files. We should verify
our panel the same way: every expected (chip, period) present, missing ones
explicitly flagged.

---

## What the new pipeline looks like (first sketch)

1. **config** – periods (months + quarters), region, model, thresholds; validated,
   frozen into each run folder.
2. **tile plan** – one deduplicated list of tiles for the region; chip IDs from
   tile key + row/col.
3. **image fetch** (CPU jobs) – one Earth Engine call per tile per period, image +
   `n_clear`/`n_total`; cached to disk; done/failed lists.
4. **detect** (GPU jobs) – read cache, score every chip, write one file per tile.
5. **segment** (GPU jobs) – SAM2 from the same cache; mask area per chip.
6. **panel** – join into one table: chip × period, probability, SAM2 area, quality
   variables, missing flags.
7. **cache cleanup** – delete imagery for a period once 4–6 are verified (if
   storage needs it).

---

## Questions for you

1. **Spatial unit.** The model's chips overlap by half (48 px chip, 24 px step), so
   each pixel is in ~4 chips. Should the panel unit be those overlapping chips, or
   a non-overlapping grid (e.g. every other chip, or 48 px cells)? Overlap matters
   for standard errors.
2. **Periods.** Which years? Monthly *and* quarterly for all of them? Do you also
   want annual?
3. **Monthly composites.** One month often has few clear images. Same recipe as
   quarterly (median, Cloud Score+ ≥ 0.6) and let `n_clear` tell you the quality,
   or something different?
4. **SAM2 prompts.** Mask only chips above a threshold in *that* period, or every
   chip that was ever above threshold in any period (so area is measured
   consistently, including zeros)?
5. **Region.** Whole Amazon (ACA), or a subset? And do you need the Andes area
   specifically?
6. **Which branch is canonical?** The driver is only on `code-walkthrough`, whose
   commit says not to develop on it. Is `main` (without the driver) the base for
   our new code?
