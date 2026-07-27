# Spatial / resolution pipeline card

Scope: this document describes **only** the production model we will use — the
**from-scratch CNN ensemble** (`48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`), run in
`embedding_strategy = none`. It does **not** use SSL4EO, foundation-model
embeddings, or the `cls_only` / `cls_patch` "dense embedding" tracks. Those exist
on the branch as a shelved experiment (see the short appendix at the end for why).

Every quantitative claim is cited to a file and line on the upstream
`2026dense_embed` branch (`earthrise-media/mining-detector`, folder `core/`).
Anything **not** established by the code is marked **[NOT IN REPO]** or
**[UNVERIFIED]**. Nothing here is invented.

---

## 0. The model in one box

| Thing | Value | Source |
|---|---|---|
| Model file | `48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` (15.8 MB) | README inference examples |
| Type | **Ensemble of CNNs trained from scratch** (CNN800k-class) | `core/README.md` "Embedding-based models"; `model_library.py:48-91` |
| Run mode | `embedding_strategy = none` (the default) | `inference_engine.py:135`; README examples pass no `--embedding_strategy` |
| Input | 48×48×13 Sentinel-2 chip (raw pixels) | `model_library.py:48`; `inference_engine.py:770-784` |
| Extra weights needed | **None** — no SSL4EO, no `.pth`, no `models/SSL4EO/` | verified: `none` path never loads an embed model (`inference_engine.py:775, 782`) |
| Detection resolution | **480 m** (48 px chip) | `tile_utils.py:16`, `inference_engine.py:130` |
| Output | GeoJSON of positive 480 m squares + `confidence`, EPSG:4326 | `inference_engine.py:635-654, 936-1003` |

Confirmed: your researcher is right that SSL4EO is not relevant. The `none` path
feeds chips straight to the Keras CNN (`inference_engine.py:782-784`); it never
touches a foundation model.

---

## 1. Deep dive — the whole process, plain language, new model only

### Region → tiles ("the shopping list")
A GeoJSON is just an **outline** — the boundary of your area, no imagery
(`inference_pipeline.py:38`). GEE can't return one giant image of the whole
region, so the code lays a grid of small squares over the outline and lists the
ones that touch it. That list of squares is the "shopping list"; each square is a
**tile** (`tile_utils.create_tiles` → `DLTile.iter_from_shape`, `tile_utils.py:66-80`).
Each tile is **552 px at 10 m/px = 5,520 m (~5.5 km)** on a side, plus a 12 px
border (`gee.py:68-69`). No imagery is pulled yet — this step only decides *which*
squares to request.

**CRS, the confusing bit.** A CRS is the rulebook for turning a spot on the round
Earth into numbers. Tiles are *measured* in **UTM (meters)**, because "10 m pixel"
and "480 m patch" are lengths and only make sense in meters (`tile_utils.py:16, 44`).
But the delivered imagery and every output file are *stored* in **EPSG:4326
(lat/lon degrees)**, the repo's standard format (`gee.py:291, 318`). Degrees are a
poor ruler (one degree of longitude shrinks toward the poles), which is exactly
why sizing is done in UTM and only the final labels are in lat/lon. Same patch of
ground, two coordinate systems doing two different jobs.

### Tile → Sentinel-2 pull inside GEE
For each tile, GEE gathers **every Sentinel-2 scene overlapping the date window**
(revisit ~5 days, so ~6 scenes/month — **[NOT IN REPO]**, general fact) and works
with **13 spectral bands** (B1–B12 + B8A; not just RGB, because bare mined earth
vs forest separates best in infrared) (`gee.py:78-79, 124`). Two preprocessing
steps happen inside GEE before anything comes back:

1. **Cloud masking, per scene, per pixel.** Google Cloud Score+ (`cs_cdf`) rates
   every pixel's clarity; only pixels with `cs_cdf ≥ 0.6` survive. This runs on
   **each scene individually** before compositing (`gee.py:72, 128-132`).

2. **Median composite.** The surviving clear pixels are stacked and, **at each
   pixel and each band independently**, the **median** value is taken
   (`gee.py:133`). Median (not mean) is chosen so a stray haze/cloud-edge outlier
   is ignored rather than dragging the value. **This date window is the only
   temporal lever** — "monthly" vs "quarterly" is just a 1-month vs 3-month window.

**⚠️ Cloud gap = silent zero (TODO: add a coverage flag).** If a pixel was cloudy
in *every* scene, nothing survives the median, and the repo does **nothing**
special: it sets no nodata for the image product (`gee.py:311`), does not count
valid pixels, and never flags or skips a low-data tile. The masked pixel comes
back as **0** (**[NOT IN REPO]** — Earth Engine default fill, worth verifying
empirically), so a fully-clouded area arrives as a **black image** and is later
classified as "no mine" — **indistinguishable in the output from a true
negative**. For a monthly time series this is the key risk. *Action item:* add our
own per-tile coverage check (fraction of non-zero pixels) and flag periods where a
tile is mostly gap. This mechanism does not exist in the repo today.

The pull itself fits the tile onto a fixed **576×576** grid
(`clipToBoundsAndScale`, `gee.py:191-195`). 576 = the 552 px tile **+ 12 px pad on
each side** (`gee.py:190`); the padding gives tiles a small overlap so nothing is
lost at a seam. What returns is a **tensor of 576 × 576 × 13**: a 576×576 spatial
grid where each cell holds a 13-number vector (one per band). The 576 and the 13
are separate axes — the 13 is *not* a sum of layers, it's the depth at each pixel.

> **[NOT IN REPO]** S2 bands are natively 10 m (B2/B3/B4/B8), 20 m
> (B5/B6/B7/B8A/B11/B12), 60 m (B1/B9/B10). Requesting them all on one 10 m grid
> means GEE resamples the 20 m/60 m bands up to 10 m — GEE default behaviour, not
> specified in this code.

### Tile → chips (48 px)
The CNN never sees the whole tile; it judges one small square at a time. So a
**48×48 px window slides across the 576×576 tile in 24 px steps** (stride =
chip_size ÷ 2 → 50 % overlap) (`inference_engine.py:130, 140, 603-614`). That
yields ~**23×23 ≈ 529 chips per tile**. Before chipping, pixels are normalized:
`clip(pixels / 10000, 0, 1)` — Sentinel-2 L1C reflectance is scaled 0–10000, so
this maps it to 0–1 for the network (`inference_engine.py:770`).

Each chip is really **two things bundled** (`cut_chips`, `tile_utils.py:150-190`):

1. the **pixels** — a 48×48×13 mini-tensor (what the model sees; 48 px = **480 m**), and
2. a **polygon** — a plain lat/lon rectangle (EPSG:4326) describing *where on Earth*
   that square sits (the location the eventual score gets pinned to).

Because of the 50 % overlap, these chip polygons **overlap each other** — they are
not a clean grid. (That's why a later dissolve step merges them before you measure
area.)

### Chip → CNN classification (`none` mode)
Each 48×48×13 chip goes straight into the **CNN ensemble** and out comes **one
number 0–1**: the probability it's mining (`inference_engine.py:782-784`). Under
the hood each ensemble member is a from-scratch convolutional network — stacked
`Conv2D(3×3)` layers building from edges/textures up to "this looks like a pit,"
ending in `Dense(1, sigmoid)` (`model_library.py:48-91`, `CNN800k`). "Ensemble"
means several such CNNs vote and their scores are **averaged** into one confidence
(`_probs_from_predictions`, `inference_engine.py:622-633`). No embeddings, no
foundation model — the network learned mining patterns directly from labeled
Sentinel-2 chips during training.

A chip is **kept** if its probability exceeds `pred_threshold` (default **0.5**;
the README's Amazon example uses 0.4); survivors get `confidence` = that
probability, attached to their polygon (`inference_engine.py:635-654`). (Recall
the cloud gotcha: an all-zero cloudy chip just scores low → "no mine.")

### Chips → detection GeoJSON
`bulk_predict` writes a cumulative GeoJSON of the positive chip polygons +
`confidence`, EPSG:4326, to `data/outputs/<model-version>/<region>…geojson`
(`inference_engine.py:936-1003`). Every detection is a **480 m square**, and they
**overlap** because of the 50 % stride.

### (Optional) SAM2 segmentation — OFF by default
Not part of the default run (`run_sam2 = False`, `inference_engine.py:137`). If
enabled, SAM2 takes the positive squares as **box prompts** on the true-colour
(B4,B3,B2) image and produces a **pixel-level (10 m) mining mask**
(`inference_engine.py:1147-1201`). This is what turns coarse 480 m squares into
fine mine-scar outlines. It needs SAM2 weights (`models/sam2/`, not in git) and
extra setup (`core/README.md` "Masking").

### (Optional) Post-processing — separate step
Run on the output GeoJSON afterward (`postprocess.py`): a **dual-threshold filter**
keeps `confidence ≥ t_main` (default 0.43) and forces isolated detections (5th
nearest neighbour > 3 km) to clear a stricter `t_iso` (default 0.75), suppressing
lone false positives; an optional **dissolve** merges overlapping squares into
polygons with mean confidence (`postprocess.py:51-70, 99-170`).

---

## 2. Resolution card (new model)

| # | Step | Spatial unit / resolution | Source |
|---|------|---------------------------|--------|
| 1 | Region input | polygon, EPSG:4326 | `inference_pipeline.py:38` |
| 2 | Tiling | **552 px @ 10 m = 5,520 m** tile (UTM sizing) +12 px pad → 576 px | `gee.py:68-69`, `tile_utils.py:66-80` |
| 3 | Sentinel-2 pull | **576×576×13** tensor, nominally 10 m/px, EPSG:4326 | `gee.py:190-195, 291, 318` |
| 4 | Cloud masking | per-scene, per-pixel, keep `cs_cdf ≥ 0.6` | `gee.py:72, 128-132` |
| 5 | Median composite | per-pixel, per-band median over date window | `gee.py:133` |
| 6 | Normalize + chip | `/10000` clip 0–1, then **48 px = 480 m** chips, 24 px stride (50 % overlap) | `inference_engine.py:770, 130, 140, 603-614`; `tile_utils.py:150-190` |
| 7 | CNN classification | one prob per chip; keep if `> pred_threshold` (default 0.5) | `inference_engine.py:782-784, 635-654` |
| 8 | Detection output | GeoJSON of **480 m** squares + `confidence`, EPSG:4326 (overlapping) | `inference_engine.py:936-1003` |
| 9 | SAM2 (optional, OFF) | upsampled to **10 m** binary mask | `inference_engine.py:137, 1147-1201` |
| 10 | Post-process (separate) | dual-threshold filter + optional dissolve | `postprocess.py:51-170` |

**In short:**
- **Before GEE:** pick sub-region → cut into ~5,520 m UTM tiles (the shopping list).
- **Inside GEE:** per tile, gather all scenes in the window → cloud-mask each → per-pixel median composite → return **576×576×13** (EPSG:4326), 12 px pad included.
- **Locally:** normalize `/10000` → slide 48 px chips at 24 px steps → CNN scores each → keep > threshold → GeoJSON of overlapping 480 m squares → (optional) SAM2 10 m masks → (optional) post-process.

---

## 3. Monthly vs quarterly

The only temporal control is the median-composite window (`gee.py:128-133`).
Because the new model runs on **Sentinel-2 median composites** (not annual
embeddings), monthly/quarterly is fully in your hands — you just set
`--start_date` / `--end_date`.

- **[IN REPO]** Each period = a median of cloud-masked scenes; strict masking
  (`cs_cdf ≥ 0.6`). Fewer clear scenes → more zero-filled (gap) pixels.
- **[NOT IN REPO]** ~5-day revisit means a month has only a handful of candidate
  scenes; over the cloudy Amazon a **monthly** composite can have large gaps,
  while **quarterly** is far more likely gap-free. Monthly = finer time resolution
  but noisier/holier; quarterly = cleaner but coarser.

**Experiment (everything needed is in the repo):** same small region, same year,
run (a) 12 monthly windows and (b) 4 quarterly windows. Compare (i) clear-pixel
coverage per tile — build the coverage flag from the cloud-gap TODO above, (ii)
stability of detected count/area, (iii) compute time. The composite step is where
the monthly signal will hold up or fall apart.

---

## 4. What changed vs your old model

| | Old (your current) | New (use this) |
|---|---|---|
| File | `48px_v0.X-SSL4EO-MLPensemble` (0.7 MB) | `48px_v4.10b-…-ensemble.h5` (15.8 MB) |
| Type | MLP probe on **frozen SSL4EO embeddings** | **CNN ensemble from scratch** |
| Mode | `cls_only` | `none` |
| Needs SSL4EO `.pth`? | Yes | **No** |
| Chip / resolution | 48 px = 480 m | 48 px = 480 m (**same**) |

They moved off the SSL4EO-embedding approach (which your v0.X model *is*) back to a
stronger from-scratch CNN — which is exactly why SSL4EO stopped mattering. Spatial
resolution is unchanged (480 m); only the model internals and the dropped
foundation-model dependency changed.

---

## 5. Open / unverified items

- **[TODO]** Cloud-coverage flag: no valid-pixel / gap detection exists in the
  repo (§1). Needed before trusting monthly runs.
- **[UNVERIFIED]** Masked-pixel fill value assumed to be 0 (EE default) — confirm
  by pulling a known-cloudy tile and inspecting the array.
- **[UNVERIFIED]** Time-series differencing / cumulatives live in
  `core/cumulatives_and_diffs.ipynb` and `core/model_evaluation.ipynb`, not yet
  read line-by-line — this is where the gold-price-correlation series would be
  built. Flag for a dedicated pass.
- **[UNVERIFIED]** `inference_pipeline.py:54` logs area as `(tilesize/100)**2`
  labeled "ha"; with 10 m pixels that's really km² (≈30.5 km²/tile). Unit-label
  quirk — confirm before trusting the logged number.

---

## Appendix — why the branch is called `2026dense_embed` (and why we ignore it)

The branch contains an experimental "dense embedding" track: a frozen **SSL4EO
ViT-S/16** foundation model turns imagery into embeddings, and a small probe
classifies them — either one embedding per 48 px window (`cls_only`, 480 m) or a
14×14 grid of 16 px = **160 m** patch-cell embeddings (`cls_patch`, finer)
(`inference_engine.py:128, 187-193, 372-405`; `embed_cls_patch.py:32-33`). The
README states plainly that **as of May 2026 the best model is still the
from-scratch CNN ensemble**, and the embedding approach "did not beat" it — it's
kept only as an alternate paradigm (`core/README.md` "Embedding-based models").
The finer 160 m resolution is tempting for granular work, but it needs the SSL4EO
`.pth` and is not the recommended model. **We are deliberately not using it.**
