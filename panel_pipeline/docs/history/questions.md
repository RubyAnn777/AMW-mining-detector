> **History, not maintained.** Ruby's own notes on overlapping chips and SAM2 (8 Oct). The reasoning is in `round2_answers.md` §2 and §7; the open questions are D4–D6 in `../open_design_decisions.Rmd`.

Overlapping chips in any kind of regression model

Reasons it might not make sense as an econometric unit:

The observations are not independent. One mine scar lights up a block of overlapping chips. A regression treats them as 4 separate findings when they are one. Your standard errors will be too small unless you cluster them, for example by tile or with Conley spatial standard errors. The effective sample size is roughly a quarter of the row count.
Built-in spatial correlation. Neighbouring chips share pixels, so their outcomes and their quality variables are mechanically correlated. Any spatial lag or spillover estimate picks this up as if it were a real effect.
It bends the postprocess variable. The isolation rule asks whether a chip's 5th-nearest detected neighbour lies within 3 km. A single scar already fires several overlapping chips, so a chip can "have 5 neighbours" just from its own scar. The rule mostly measures blob size, not clustering of mines.
Areas add up wrong. If you sum SAM2 area over chips, every pixel is counted up to 4 times. Totals must come from a non-overlapping partition.
Size. The panel is about 4 times larger than a non-overlapping one (§9).

Reasons to keep it, which I think win for the dataset:

It is how the detector was designed and validated, so every chip is a case the model was built to score.
It keeps every score, so nothing is lost.
It can be undone. Taking every other chip in both directions gives an exact non-overlapping 480 m grid. One detail: each tile row holds 23 chips, an odd number, so "every other chip" must be counted globally (tile index × 23 + chip index), not inside each tile. Otherwise the pattern flips at every tile edge.


SAM2 and overlap
What their SAM2 actually does, step by step
Prompt set. A detections file: Amazon loose set, or raw Andes set.
Where to look. _simplify_for_tiling (inference_engine.py:1555) blurs the detections into blobs (buffer of 0.005°, about 550 m, out and back in, holes removed). Then 256 px tiles (+12 px padding) are laid over the blobs (sam2_mask.py:184, so not the detector's 552 px tiles).
Imagery. Each 256 px tile fetches its own composite from Earth Engine (same recipe, same dates).
The prompt. polygon_gdf_to_pixel_bbox (inference_engine.py:1342) makes one single box per tile: the bounding box around all detected chips in that tile, not one box per chip.
SAM2. It runs on an RGB picture (bands 4, 3, 2 divided by 3000) with that box. It returns 3 candidate masks, and the one with the best score is kept.
Spatial prior. A penalty is added that is 0 inside detected chips and grows with the square of the distance outside them (σ = 12 px). The result is smoothed and cut at 0, giving the mask (-msk.tif). The smoothed scores are also saved, capped at ±16 (-logits.tif).
No restart and no failure record. It doesn't skip finished tiles, and failed tiles are only printed (inference_engine.py:1655).
Correction to what I said last time

I suggested running SAM2 on the detector's 552 px tiles so the imagery could be reused. That was wrong, so I'm withdrawing it. Because the prompt is one box around everything detected in the tile, tile size changes the answer. On a 552 px tile the box can cover several separate mines and most of the tile, so the result would be a different, untested setting. The 256 px tiles are part of how their masks are made.

And the extra download is small. Their notes put a basin-wide SAM2 mask at roughly 15,000 tiles of 280 px for one annual period (docs/design/persistence-planning.md:118), against about 270,000 detector tiles of 576 px (my estimate for Amazon ACA, see §9). That is about 1–2% of the detector's pixels. Keeping their design costs little and keeps their results.

My proposal

Per period (each month and each quarter separately), after that period's detections are finished:

Step	What	Same as theirs?
a. Prompt sets	(1) Amazon: chips passing the loose dual threshold (≥ 0.4 raw, then 0.43/0.75). (2) Andes: in_andes chips with mean ≥ 0.2	Yes, the same as their annual masks
b. Tiles and prompts	_simplify_for_tiling, 256 px tiles, one box per tile, prior, smoothing: unchanged	Yes
c. Two runs	Amazon set and Andes set masked separately, then combined pixel by pixel (mine if either says mine)	Yes, the same as their two runs and merged mosaic
d. Imagery	Fetch only bands 4, 3, 2 plus n_clear for SAM2 tiles	Same pixel values (SAM2 only reads those 3 bands), about 4× smaller download and cache
e. Restart	Skip tiles whose mask and logits exist; write a failed-tiles list; retry	New
f. Per-chip numbers	For every chip inside a SAM2 tile, detected or not: masked pixels, observed pixels (n_clear > 0), mask area in m², mean and max logit, and whether any SAM2 tile covered it	New: this is what goes into the panel

Two points that matter for the econometrics:

Why every chip in a SAM2 tile, not just detected ones: their notes say SAM2 often segments scars that extend past the chip that prompted them (persistence-planning.md:99). Counting mask pixels in every chip's footprint captures that extent in the neighbouring chips.
SAM2 area is only measured where detection fired. A chip outside every SAM2 tile gets area 0 by construction, not because it was measured as 0. So the panel needs a sam2_covered flag, and area should be read together with it. This is a selection issue your analysis must handle.
Monthly cloud gaps. Pixels with no clear image come back as 0 in all bands (black). SAM2 sees them as black holes. So area must be reported over observed pixels, and the observed share is itself a quality variable.

Combining overlapping SAM2 tiles. Neighbouring SAM2 tiles overlap in their 12 px padding. Their mosaic step combines overlaps as "mine if any tile says mine" (sam2_build_cog.py, build_mask_union_vrt). We use the same rule when counting pixels per chip.

Counting pixels per chip cheaply. Chips and mask pixels are both upright rectangles in latitude/longitude. So for each SAM2 tile we can build a "running total" table of the mask (each cell holds the sum of everything above and to the left of it). Then the pixel count inside any chip takes 4 lookups, whatever the chip's size. This is called a summed-area table. I'll explain it properly when we write that file.