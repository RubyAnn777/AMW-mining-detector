<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# Decision Register — every parameter, choice, and knob in the pipeline

Purpose: single reference for all decisions we can make — methodological,
parametric, operational. Ordered along the process (design → data → model →
post → time series → analysis → compute). Each entry: where it is set, current
value, admissible range, and consequence. Voice: dense, minimal prose.

**Status codes** — `SET` we chose it deliberately · `DEFAULT` inherited upstream
default, changeable · `FIXED` changing requires code changes / retraining ·
`OPEN` decision pending, evidence expected from the pilot.

Cross-references: model details `pipeline_spatial_card.md`; code explanation
`timeseries/LECTURE.md`; upstream mechanics `CODE_WALKTHROUGH.md`.

---

## A. Study design (methodological)

| # | Decision | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| A1 | Unit of analysis | `panel.py` (chip grid) | 480 m chip, 240 m stride (overlapping) | chip level; dissolved polygons; tile (5.5 km); pixel (10 m, needs SAM2) | Chip = model-native, balanced panel, overlap must be dissolved before area | SET |
| A2 | Outcome variable | `panel.py` → `prob` | continuous CNN score ∈ [0,1] | binary (thresholded); continuous; area after dissolve | Continuous preserves information; threshold late (see H6) | SET |
| A3 | AOI (pilot) | `data/boundaries/la_pampa_corridor.geojson`; `--region` | La Pampa corridor ~1,900 km², ~70 tiles | any GeoJSON; existing: `madre_de_dios_mining_aoi` (679 tiles), test regions, basin (219,800 tiles) | Sample size & runtime scale linearly in tiles; AOI chosen non-random (dense signal) → no external validity | SET (pilot) |
| A4 | Study year | `--year` | 2025 | any year with S2 (2016+; Cloud Score+ from 2016) | 2025 = complete, comparable to upstream outputs (through 2025Q3) | SET |
| A5 | Cadence | `run_trials.py` (both run) | monthly AND quarterly, compared | monthly / quarterly / annual / rolling windows / hybrid (quarterly anchor + monthly interpolation) | **The pilot's core question.** Decision after REPORT.md sections A, E–G | OPEN |
| A6 | Period boundary convention | `month_specs`/`quarter_specs` | calendar months/quarters, end-exclusive | hydrological year; rolling 30-day; scene-aligned windows | End-exclusive matches `ee.filterDate`; calendar aligns with econ covariates (gold price) | SET |
| A7 | Ground truth / validation | — | none yet | AMW published quarterly outputs; manual photo-interpretation sample; MAAP/RAMI products | Without it, all pilot results are internal-consistency only | OPEN |
| A8 | Scale-up path | project plan | corridor → Madre de Dios → basin | any | Gate each step on pilot cost + noise results (§J worked example) | SET |

## B. Data acquisition — the GEE request

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| B1 | Collection | `DataConfig.collection`; `--collection` | `S2L1C` (COPERNICUS/S2_HARMONIZED) | `S2L1C-12band`, `S2L2A` (surface refl.), `S1` (radar), `AlphaEarth` (annual embeddings) | **Model was trained on S2L1C 13-band — changing collection invalidates the model.** S1 is cloud-immune → future wet-season option (pipeline exists: `core/gee_S1_pipeline.ipynb`) but needs its own model | FIXED (for this model) |
| B2 | Bands | `DataConfig._BAND_IDS` | 13: B1–B12 + B8A, order B1…B7,B8A,B8,B9,B10,B11,B12 | 12-band variant drops B10 | Order is a training contract — never reorder | FIXED |
| B3 | Date window | `--start_date` / `--end_date` | per period spec | any; end-exclusive | The only temporal lever; window length ⇒ scenes per composite (~6/month) | SET per run |
| B4 | Tile size / pad | `DataConfig.tilesize`, `.pad`; `--tilesize --pad` | 552 + 12 px pad → 576 px request | any; 576 must stay divisible by stride (24) | Pad gives seam overlap; larger tiles = fewer, bigger requests (EE memory-limit risk) | DEFAULT |
| B5 | Resolution | `create_tiles(resolution=10)` | 10 m/px | 10/20/60 native S2 | 20/60 m bands upsampled by GEE to 10 m grid (default resample) | FIXED |
| B6 | Request grid | `clipToBoundsAndScale(width,height)` | 576×576 | `scale=` alternative | Fixed grid guarantees array shape = coverage raster shape | FIXED |
| B7 | EE endpoint | `gee.py:45` | `earthengine-highvolume.googleapis.com` | standard endpoint | High-volume = built for many small programmatic requests | DEFAULT |
| B8 | Concurrent GEE requests | `DataConfig.max_workers`; `--max_workers` | 8 | 1–~40 (EE quota-limited) | Throughput vs 429 rate-limit errors; also panel.py batch size | DEFAULT |
| B9 | Per-request retry | `@retry.Retry(timeout=240)` on `get_tile_data`; batch retry pass in `get_tile_data_concurrent`; `tries` in engine | 240 s; 1 batch retry; 2 tries | code edit | Resilience to transient EE failures; failed tile → None → recorded FAILED, never zero | DEFAULT |
| B10 | Image cache | `DataConfig.image_cache_dir`; `--image_cache_dir`; `--cache_dir` (trials) | `data/timeseries/cache/2025` | any dir / off | Re-runs free; disk ~8.5 MB/tile/period (uint16 deflate); cache key = collection+tile+window | SET |
| B11 | EE auth mode | env `EE_PROJECT`; optional `GOOGLE_APPLICATION_CREDENTIALS` | user auth, project `arboreal-…` | service-account JSON (for headless VMs) | Service account needed if runs move to a cloud VM (§J) | DEFAULT / OPEN at scale |
| B12 | Request architecture backlog | `OPTIMIZATIONS.md` | 2 requests/tile-window (image + coverage) | O1 merge into 1 request; O2 derive coarse coverage by summation; O3 larger tiles; O4 two-tier cadence; O5 batch export | See `OPTIMIZATIONS.md` for impact/effort/risk per item | PLANNED |

## C. Cloud masking & compositing (inside GEE)

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| C1 | QA source | `gee.py _build_composite` | Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) | s2cloudless; QA60 bitmask; Fmask | CS+ is Google's current best; per-pixel continuous score | DEFAULT |
| C2 | QA band | `QA_BAND` | `cs_cdf` | `cs` (raw score) | `cs_cdf` = CDF-calibrated, recommended | DEFAULT |
| C3 | Clear threshold | `DataConfig.clear_threshold`; `--clear_threshold` | 0.60 | 0.50–0.65 (Google guidance); any [0,1] | ↑ = stricter mask, fewer clear px, more gaps; ↓ = haze leaks into composite. **Must match between imagery and coverage pull** (enforced in `panel.py`) | DEFAULT |
| C4 | Composite operator | `_build_composite` | `median()` (S2); `mosaic()` (S1) | mean, percentile(p), medoid, best-available-pixel — code edit | Median = outlier-robust; nonlinear ⇒ quarterly ≠ mean of monthlies (analysis §E caveat) | FIXED |
| C5 | Gap fill value | EE `computePixels` behavior | masked → 0, all bands | none (transport-fixed) | The silent-zero problem; flagged via C6, verified nightly (`fill0_mismatch`) | FIXED (monitored) |
| C6 | Coverage counting (ours) | `coverage.py` | `N_CLEAR` = count of unmasked scenes (B2, masked stack); `N_TOTAL` = count (raw stack); `unmask(0).toUint16()` | count on other band; per-band counts | Chip flag `clear_frac = mean(N_CLEAR>0)`; `n_clear` = median sample size (precision) | SET |

## D. Tiling → chips → normalization (local)

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| D1 | Tiling scheme | `tile_utils.create_tiles` → DLTile | DL global UTM grid | `CenteredTile` (custom centers) | DLTile = deterministic ⇒ stable tile keys ⇒ stable chip_ids across periods | FIXED |
| D2 | Chip size | `InferenceConfig.geo_chip_size`; `--geo_chip_size` | 48 px = 480 m | model-fixed (input layer) | Detection resolution; changing = retraining | FIXED |
| D3 | Stride | `stride_ratio`; `--stride_ratio` | 2 → stride 24 px (50 % overlap) | 1 (no overlap, 4× fewer chips, coarser); ≥3 (denser, slower) | Overlap smooths detection maps; 576 must divide by stride | DEFAULT |
| D4 | Normalization | `predict_on_tile_pixels`; mirrored in `panel.py` | `clip(px/10000, 0, 1)` | training contract | Must match training bit-for-bit | FIXED |

## E. Model & inference

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| E1 | Model | `--model_path` / `--model` | `48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` (CNN ensemble, from scratch) | archived: SSL4EO-MLP v0.X (`cls_only`); `cls_patch` dense-embed track (160 m, needs SSL4EO `.pth`); future upstream releases | v4.10b = upstream's current best; embedding track shelved by upstream ("did not beat") | SET |
| E2 | Embedding strategy | `--embedding_strategy` | `none` | `cls_only`, `cls_patch` | `none` = raw-pixel CNN, no extra weights, no FM | SET |
| E3 | Prediction threshold | `--pred_threshold`; `--threshold` (trials); `T_MAIN` (analysis) | 0.4 (config-t0.4; engine default 0.5) | [0,1]; for the CURRENT ensemble only t0.4 and t0.43 exist in `models/` (t0.39/t0.5 belong to the superseded `...-20cg-21a-ensemble` in `_archive/models/`) | Sensitivity/specificity trade; panel stores all probs so this only gates the convenience GeoJSON; analysis sweeps 0.30–0.60 (H5) | SET (provisional) |
| E4 | Ensemble aggregation | `_probs_from_predictions` | mean over members | vote, max, trimmed mean — code edit | Mean = upstream convention | FIXED |
| E5 | Concurrency (tiles in flight) | `max_concurrent_tiles`; `--max_concurrent_tiles` | 500 (upstream pipeline); 8-tile batches (our panel) | 1–∞, RAM-bound (~17 MB/tile) | Our batching bounds memory; upstream value only matters if using `bulk_predict` | SET |
| E6 | Retries per tile | `tries`; `--tries` | 2 | ≥1 | With B9, three-layer resilience | DEFAULT |
| E7 | Output form | upstream `bulk_predict` vs our `panel.py` | panel: ALL chip probs, parquet per period | upstream: positives-only cumulative GeoJSON | Panel is the analysis dataset; upstream form only for maps | SET |

## F. SAM2 segmentation (optional stage — currently OFF)

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| F1 | Enable | `run_sam2`; `--run_sam2` | False | True | 480 m squares → 10 m scar masks; needs weights + GPU/MPS; excluded from pilot & cost model | SET (off) |
| F2 | Checkpoint | `MaskConfig.sam2_checkpoint` | `models/sam2/sam2.1_hiera_small.pt` | other SAM2 sizes (tiny→large) | Quality vs VRAM/time | DEFAULT |
| F3 | Fine-tuned weights | `finetuned_weights` | `SAM_model_96_px_final.pth` | — | AMW's mining fine-tune | DEFAULT |
| F4 | Hydra config | `sam2_model_cfg` | `configs/sam2.1/sam2.1_hiera_s.yaml` | must match F2 | — | DEFAULT |
| F5 | Spatial prior σ | `prior_sigma`; `--prior_sigma` | 12.0 px | >0 | Falloff of box-prompt prior | DEFAULT |
| F6 | Smoothing σ | `smoothing_sigma`; `--smoothing_sigma` | 2.5 px | ≥0 | Gaussian smoothing of upsampled logits | DEFAULT |
| F7 | Mask output dir | `mask_dir`; `--mask_dir` | `data/outputs/sam2` | any | — | DEFAULT |

## G. Post-processing (separate step, on detection GeoJSONs)

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| G1 | Main threshold | `postprocess.py --t-main` | 0.43 | [0,1] | Final keep threshold (matches config-t0.43) | DEFAULT |
| G2 | Isolation threshold | `--t-iso` | 0.75 | [0,1] | Stricter bar for isolated detections → kills lone false positives, also kills new frontier pits (bias vs noise trade) | DEFAULT |
| G3 | Neighbor rank | `--neighbor_rank` | 5 (5th NN) | ≥1 | Defines "isolated" | DEFAULT |
| G4 | Isolation distance | (arg, default) | 3.0 km | >0 | with G3 | DEFAULT |
| G5 | Dissolve | `--dissolve` flag | off unless passed | on/off | Merges overlapping chips → polygons, mean confidence; required before any area statistic | SET (on, in analysis) |

## H. Time-series construction & analysis (our layer)

| # | Parameter | Where set | Current | Options / range | Consequence | Status |
|---|---|---|---|---|---|---|
| H1 | Observation gate τ | `analysis.py TAU_OBSERVED` | 0.95 | (0,1]; sensitivity 0.50 reported | Chip counts as observed iff `clear_frac ≥ τ`; ↑τ = cleaner but more missing | SET (provisional) |
| H2 | Missingness treatment | analysis design | observed-only series + naive shown side by side | listwise deletion; imputation; state-space gating | Pilot quantifies; final choice = econometric stage (H7) | OPEN |
| H3 | Area measure | `dissolved_area_km2` | union of chips, EPSG:32719 | sum×correction; count×cellarea | Union avoids 4× overlap double-count; UTM19S correct for MdD (re-decide zone at basin scale → use equal-area, e.g. EPSG:6933) | SET |
| H4 | Threshold grid | `T_GRID` | 0.30–0.60 step 0.05 | any | Sensitivity sweep (§G of report) | SET |
| H5 | Near-threshold band | `NEAR_BAND` | ±0.05 around 0.4 | any | Fragility metric | SET |
| H6 | Threshold timing | design | threshold LAST (continuous probs through pipeline) | threshold early | Early thresholding converts score noise → area noise | SET |
| H7 | Interpolation model (final TS) | not yet coded | — | linear interp; Kalman/state-space with obs. variance ∝ 1/n_clear and gate on clear_frac; quarterly anchors + monthly disaggregation; GAM/seasonal | THE downstream decision; pilot report §I gives the evidence | OPEN |
| H8 | n_clear precision bins | `section_nclear` | [1,2,3,5,8,∞) | any | Tests precision mechanism → justifies weights in H7 | SET |
| H9 | Balanced-panel differencing | analysis design | Δ only on co-observed chips | naive Δ | Removes composition artifact (report §D) | SET |
| H10 | Monthly→quarterly aggregation (for comparison) | `section_consistency` | mean of 3 months; any-positive; both reported | max, median | Different noise behavior; report reads both | SET |

## I. Statistical inference choices

| # | Choice | Current | Alternatives | Rationale | Status |
|---|---|---|---|---|---|
| I1 | Correlation | Spearman ρ | Pearson | rank-based, robust, honest at n≈12 | SET |
| I2 | Group comparison | Mann-Whitney U | t-test | no normality assumption | SET |
| I3 | Agreement | Cohen's κ + IoU | raw % agreement | chance-corrected + spatial | SET |
| I4 | Uncertainty reporting | descriptive only, no stars | bootstrap CIs (possible follow-up) | n=12 periods; bootstrap over chips ignores spatial correlation → would need block bootstrap by tile | SET (provisional) |
| I5 | Spatial autocorrelation handling | none (pilot) | block bootstrap by tile; Moran's I diagnostics | chips overlap + mining clusters ⇒ effective n ≪ chip count; flag in any write-up | OPEN |

## J. Compute & operations

| # | Decision | Current | Options | Trade-off | Status |
|---|---|---|---|---|---|
| J1 | CPU vs GPU | CPU (Mac) | GPU/MPS | Detection CNN is tiny (48×48×13); bottleneck is **network wait on GEE**, not FLOPs → GPU near-useless for detection. GPU/MPS matters only if SAM2 (F1) enabled | SET |
| J2 | Compute location | local MacBook | GCP VM (e.g. `n2-standard-32`, ≈$1.55/h); university HPC; other clouds | VM: closer to EE, no sleep risk, needs service-account auth (B11) + code deploy; Mac: free, fine at pilot scale | SET (pilot) / OPEN (scale) |
| J3 | Parallelism | 8 fetch threads (B8), 1 inference thread | ↑workers; multiple processes over tile shards | EE quota + RAM bound; beyond ~16 workers expect 429s | DEFAULT |
| J4 | Wall-time model | `time_tiles.py` measures s/tile (cold/warm/infer) | — | core-hours = tiles × periods × s_tile/3600. Prior obs: 679 tiles ≈ 2 h @ 8 workers ⇒ ~10.6 s/tile wall. Pilot: 70 × 16 × 10.6 s ≈ 3.3 h + ~30 % coverage fetches. Basin quarterly: 219,800 × 4 × 10.6/3600 ≈ 2,590 core-h ⇒ ~81 h on 32 procs ≈ $130/yr at J2 pricing (order-of-magnitude; re-estimate from tonight's timings CSV) | OPEN (data tonight) |
| J5 | Storage | cache ~8.5 MB/tile/period | prune cache; no cache | Pilot ≈ 10 GB; basin-monthly ≈ 22 TB/yr → at scale, cache OFF or lifecycle-managed bucket | OPEN |
| J6 | Sleep prevention | `caffeinate -im -w <PID>` in launcher | VM (moot) | lid must stay open; power connected | SET |
| J7 | Resume strategy | period-level skip (parquet exists) + tile cache | none | crash cost ≈ one period | SET |
| J8 | Scheduling | manual overnight | cron/launchd; scheduled cloud jobs | manual fine at pilot cadence | SET (pilot) |
| J9 | Failure policy | per-tile → FAILED row; per-period → logged, continue; exit code ≠0 → analysis not run | abort-on-first-error | maximizes usable data per night | SET |

## K. Environment & reproducibility

| # | Item | Current | Notes | Status |
|---|---|---|---|---|
| K1 | Python env | `venv/`, lock `requirements-lock_2026-07-17.txt` | re-lock after any install | SET |
| K2 | Upstream pin | `upstream/main` @ `c473933`; asserts in `panel.py` guard private-API drift | re-pin on every sync (GIT_PLAYBOOK routine) | SET |
| K3 | Output versioning | outputs keyed by model version + period label; panels immutable once written | consider committing panels (small) but not cache | SET |
| K4 | Analysis constants | all module-level in `analysis.py` (H1–H10) | change → re-run analysis only, data untouched | SET |
| K5 | Randomness | none in inference (deterministic model, deterministic tiling) | GEE composite deterministic given window | — |
| K6 | Secrets | `.env` (gitignored); parsed, never sourced | never commit; service-account JSON likewise | SET |

---

## The open-decision shortlist (what actually needs deciding, in order)

1. **A5 cadence** — after tonight: monthly, quarterly, or hybrid. Evidence: report §§A, C–G.
2. **H7 interpolation model** — jointly with A5; evidence §§D, F, H, I.
3. **H1 τ and E3 threshold** — set from sensitivity sections, then freeze.
4. **A7 validation source** — needed before any external claim.
5. **J2/J4/J5 scale-up compute** — re-cost from tonight's `tile_stats_*.csv`; decide provider + storage policy before Madre-de-Dios-wide run.
6. **B11 auth** — service account, iff J2 moves to a VM.
7. **I5 spatial autocorrelation** — before writing up significance of anything.
