<!-- PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector -->
# Optimization Backlog — Earth Engine request architecture & scaling

Status of scaling optimizations: done, planned, and considered-but-rejected.
Companion to `DECISION_REGISTER.md` (§B, §J). Voice: dense, minimal prose.
Impact figures assume basin scale (~220k tiles) unless noted.

**Status codes** — `DONE` implemented · `PLANNED` do before next scale-up ·
`EXPERIMENT` measure first · `CONTINGENCY` only if quotas force it · `REJECTED` ·
`GUARDED` hazard with a code-level backstop in place.

## Index

| § | topic | status |
|---|---|---|
| D1–D7 | request-architecture baseline already in place | `DONE` |
| O1 | merge coverage into the imagery request (−50 % requests) | `PLANNED` |
| O2 | derive coarse-cadence coverage by summing monthly | `PLANNED` |
| O3 | larger tiles (1152 px) | `EXPERIMENT` |
| O4 | two-tier spatial cadence at basin scale | `PLANNED` |
| O5 | batch export architecture | `CONTINGENCY` |
| M1–M7 | minor / rejected, incl. **M7 streaming panel writes** | mixed |
| **M3-revised** | **inference is the bottleneck, not network — measured** | `MEASURED` |
| **P1 / P1a** | **overlapping AOIs duplicate chips; polygon-disjoint is not enough** | `GUARDED` |
| **P1b** | **the deduplication recipe** | `USE THIS` |
| **P2** | **`(H,W,B)` vs `(B,H,W)` axis order — silent corruption** | `GUARDED` |
| **P3** | **stop storing chip geometry; derive it (32 % of the panel)** | `PLANNED` |
| **P4** | **`EE_PROJECT` falls back to upstream's project — silent** | `HAZARD` |
| **P5** | **`float32`→`uint16` cache round-trip; cold vs warm may differ** | `UNVERIFIED` |
| **P6** | **all-zero imagery tile passed as fully observed (found in run 1)** | `GUARD NOW` |
| **P7** | **cache: no lifecycle, cost negligible, check `gs://amw-dev/` first** | `ACTION` |

**Project constants, assumed permanent** (a change would invalidate P1b and every cached
tile): `tilesize=552`, `pad=12`, `resolution=10 m`, `chip_size=48`, `stride=24`,
`clear_threshold=0.6`, band set = 13. `tile.key` encodes the first three, so
`chip_id = tile_key:cN` is a permanent identifier for a patch of ground.

## Already implemented (baseline; cited in the GEE uplift application)

| ID | Optimization | Where | Effect |
|---|---|---|---|
| D1 | All compositing server-side (filter, CS+ mask, median); client gets final composite only | `core/gee.py _build_composite` | No per-scene transfer; 1 request/tile/window |
| D2 | Small bounded requests: 576 px, 13 bands, uint16 ≈ 8.5 MB | `core/gee.py`, `DataConfig` | Under memory limits; cheap retries |
| D3 | Persistent tile cache keyed (collection, tile, window) | `image_cache_dir`; `timeseries` cache | Each pixel requested ≤ once, ever; crash-proof |
| D4 | Concurrency cap (8 workers) + lazy singleton `ee.Initialize` | `DataConfig.max_workers`; `gee.py` | No 429 storms; no redundant handshakes |
| D5 | Bounded exponential retries (240 s cap) + one deferred second pass; failures isolated, never hot-looped | `gee.py` | Polite under server stress |
| D6 | Deterministic tile grid; resume-safe orchestration (period-level skip) | DLTile; `run_trials.py` | No duplicate or repeated requests |
| D7 | Per-tile cost measured before scaling | `time_tiles.py`; `tile_stats_*.csv` | Uplift request based on measurement |

## O1 — Merge coverage into the imagery request  `PLANNED` (before Madre de Dios run)

*Now:* 2 `computePixels` calls per tile-window (13-band composite + 2-band counts).
*Then:* 1 call, 15 bands — composite `.addBands(N_CLEAR).addBands(N_TOTAL)`;
client slices `arr[...,:13]` / `arr[...,13:]`.

- Impact: **request count −50 %** (basin-year: ~1.8 M → 0.9 M).
- Correctness bonus: imagery and counts built from the identical server-side
  collection in one expression → `clear_threshold` match enforced by construction.
- Implementation: subclass `GEE_Data_Extractor` in `timeseries/` (override
  `_build_composite`, extend band list) — keeps the no-edits-to-`core/` rule.
  `panel.py` splits the array; `coverage.py` retained for standalone use.
- Costs: new 15-band cache format (invalidate or version cache names, e.g.
  prefix `S2L1C+COV_`); count bands are uint16 like reflectance — no dtype issue.
- Care: bands request order defines slice order — assert band names.

## O2 — Derive quarterly (and any coarser) coverage by summing monthly  `PLANNED`

Counting is additive over disjoint windows; calendar months exactly partition
quarters/years (end-exclusive convention). Hence per pixel:
`N_CLEAR_Q = Σ N_CLEAR_months`, same for `N_TOTAL`. Exact, not approximate.

- Impact: eliminates all coverage requests at cadences coarser than monthly
  (with O1: quarterly runs need imagery-only requests).
- Does NOT apply to imagery: median is nonlinear; a quarter's composite is not
  a function of monthly composites — must always be fetched.
- Implementation: small helper in `coverage.py` (read 3 cached monthly
  GeoTIFFs, sum, write quarterly raster); `panel.py` prefers derived raster
  when the months exist.
- Care: only valid if collection, filters, and `clear_threshold` are identical
  across the summed months — assert from cache metadata/filenames.

## O3 — Larger tiles  `EXPERIMENT`

576 px ≈ 8.5 MB vs ~50 MB `computePixels` ceiling → 1152 px would give 4× area
per request, −75 % request count. Risks: 4× server-side work per request (EE
user-memory-limit errors — likely why upstream chose 576), 4× waste per failed
request, RAM 4×/tile locally. Action: measured A/B on ~20 tiles (error rate,
s/tile, EE failures) before adopting. Constraint: edge must stay divisible by
stride 24 (1152 ✓).

## O4 — Two-tier spatial cadence at basin scale  `PLANNED` (design decision, register A8/A5)

Monthly cadence only within a priority mask (buffer around historical
detections + navigable rivers); annual sweep elsewhere to catch new frontiers.
Impact: steady-state request volume −80…90 %. Cost: delayed detection of novel
frontier mining — a documented surveillance bias, revisited each annual sweep.
Implementation: priority mask from AMW published cumulative detections; two
period lists in `run_trials.py`.

## O5 — Batch export architecture  `CONTINGENCY`

If granted quotas still bind: one `Export.image.toCloudStorage` (COG) per
period for the whole AOI — few large batch tasks instead of millions of
interactive requests; read tiles locally from the COGs. Different quota pool,
different failure modes, major rearchitecture (task monitoring, storage costs
~22 TB/yr basin-monthly). Only if O1–O4 + uplift are insufficient.

## Minor / rejected

| ID | Idea | Verdict |
|---|---|---|
| M1 | Coverage at 20 m instead of 10 m (−75 % coverage bytes) | REJECTED for now — pixel-exact masks required by the interpolation strategy (register H7); revisit if bandwidth ever binds |
| M2 | Compressed GeoTIFF transport instead of raw NUMPY (−~40 % bytes) | Optional; bytes are not the binding constraint, requests are |
| M3 | Cross-tile CNN batch inference | ~~REJECTED — inference is not the bottleneck (network wait is)~~ **REOPENED 2026-08-07 — the premise was wrong, see M3-revised below** |
| M4 | Fetch per-scene imagery, composite locally (enables any operator) | REJECTED — 6–18× transfer volume; defeats D1 |
| M5 | Reconstruct quarterly imagery from monthly composites | REJECTED — median nonlinearity; methodologically invalid |
| M6 | Drop B10 (13→12 bands) | REJECTED — band set is a training contract (register B2) |

## M3-revised — inference IS the bottleneck  `MEASURED 2026-08-07`

M3 was rejected on the premise that network wait dominates. **That premise is false.**
Measured over all 1,280 per-tile records in `data/timeseries/tiles/tile_stats_*.csv`
(run 1, 16 periods × 80 tiles):

| period type | fetch s/tile | infer s/tile | inference share of wall time |
|---|---|---|---|
| healthy monthly (m2025-01…10) | 2.7–2.9 | 5.9–7.3 | **68–72 %** |
| all periods pooled | 5.29 | 21.01 | **80 %** |

Note the asymmetry that makes this easy to misread: `fetch_s` is written *amortised*
(`panel.py:224` divides the batch's wall time by the batch size) because fetches run
concurrently, while `infer_s` is per tile and **serial** — inference holds
`_tf_model_lock` (`core/inference_engine.py:619`), so it never overlaps. Per batch of 8
tiles: ~22 s of concurrent fetching against ~50 s of serialised inference.

**Second finding, from the same data — inference time tripled over one overnight run:**

| periods (chronological) | mean infer s/tile |
|---|---|
| m2025-01 … m2025-10 (20:55–23:33) | ~6.2 |
| m2025-12, q2025Q1…Q4 (05:43–10:14) | ~17.0 |

Identical work — same 529 chips, same model, same tile count. A ~2.7× slowdown after
~8 hours of continuous load is consistent with sustained thermal throttling, though not
proof (the degraded-network period m2025-11 sits between the two groups, and the machine's
state was not instrumented). It is direct evidence for the duty-cycle design in
`RUN2_SETUP_AND_AUDIT.md` §3.5: **the "12 min per month" baseline only holds on a cool
machine.**

Implications, in order of value:

1. **Any inference speedup is worth ~3× more than an equivalent network speedup**, which
   inverts the assumed priority of O1/O2 versus inference work — those remain worth doing
   for *quota* reasons (request count), just not for wall clock.
2. **`core/tf_darwin.py:26` calls `tf.config.run_functions_eagerly(True)`** at import
   (`inference_engine.py:40`), disabling TensorFlow graph optimisation for the whole
   process. It exists to dodge a Metal/tf.data kernel crash on macOS. Nobody has tested
   whether that crash still occurs on TF 2.14 + the current OS. If it does not, removing it
   is a free speedup on the dominant cost. **Test before the basin run, not during.**
3. Cross-tile batching (the original M3) is now plausibly worthwhile: `_model_infer` is
   called once per tile with 529 chips; batching several tiles per `predict` call would
   amortise Keras call overhead. Measure first — the 529-chip batch may already saturate.

Do not confuse this with the thread caps. `AMW_THREADS=4` deliberately slows inference to
protect the hardware; that is a chosen trade, and these numbers are what it trades against.

## M7 — Stream panel rows to disk instead of accumulating  `PLANNED` (before basin scale)

*Now:* `panel.py:155` accumulates every chip row for a period in a Python list and writes
one parquet at the end (`panel.py:236`). At this AOI that is 42,320 rows ≈ tens of MB —
irrelevant. At basin scale (~220k tiles) it is ~116 M rows per period and will not fit.

Upstream has the same shape of problem in a different place: `bulk_predict`
(`core/inference_engine.py:990-1003`) also accumulates a GeoDataFrame across all batches,
and additionally rewrites the *entire* output GeoJSON after every batch — O(n²) write
volume. It gets away with it because it only retains chips above threshold; the panel
retains all of them, roughly 80× more rows.

*Then:* write one parquet fragment per tile batch into `panels/<label>/part-*.parquet`, or
use a `pyarrow.ParquetWriter` opened once per period and fed row groups. Either makes
memory O(batch) instead of O(period), and has a second benefit that matters today:
**a crash would stop losing the whole period's inference** (`RUN2_SETUP_AND_AUDIT.md` §5.2).

- Care: `analysis.py:94` globs `panel_*.parquet`; a directory-per-period layout needs
  `**/*.parquet` or a post-run concat step.
- Not urgent at La Pampa scale. Required before Madre de Dios or basin.

## P1 — Panel integrity under multiple AOIs  `GUARDED 2026-08-07` — **read before any O4 work**

**The hazard.** `tile_utils.create_tiles` uses `DLTile.iter_from_shape`, and DLTile keys are
a **global UTM grid** (`552:12:10.0:<zone>:<x>:<y>`) — a key identifies ground, not a
request. `panel.py` then builds `chip_id = f"{tile.key}:c{i}"` with `i` fixed by tile
geometry. Therefore **two AOIs that overlap produce byte-identical chip_ids with identical
probabilities for the shared ground.** Pool their panels and that ground is counted twice:
every count, share, mean and κ built on it is inflated, with no error and no obvious
symptom.

**This is not hypothetical.** Verified 2026-08-07:

```
la_pampa_corridor      [-70.10, -13.10, -69.65, -12.75]
madre_de_dios_mining   [-70.75, -13.35, -69.10, -12.30]
la_pampa WITHIN madre_de_dios : True   (100 % of la_pampa area)
```

All 80 La Pampa tiles would recur in a Madre de Dios run. The natural next scale-up is
exactly this pair.

**What is NOT a problem — measured, contrary to first impression.** Adjacent tiles overlap
by 24 px (`pad=12` per side), which suggests duplicated chips at tile seams. Exact
chip-to-chip geometry over `panel_m2025-06.parquet` (42,320 chips, 80 tiles):

| cross-tile chip pairs | count |
|---|---|
| IoU > 0.99 (same ground, true duplicate) | **0** |
| IoU 0.5–0.99 | **0** |
| IoU 0–0.5 (ordinary stride-24-on-48 px overlap) | 38,244 |

Because `552 = 23 × 24`, every tile's chip grid lies on the same global 24-px lattice with
non-overlapping index ranges: tile A's last chip starts at 528, tile B's first at 552.
Adjacent, never coincident. **Within one AOI the panel is clean.** (An earlier estimate of
"14.8 % duplicated" was an artefact of using axis-aligned bounding boxes for UTM-aligned
tiles — corrected here.)

**Guard implemented.** `analysis.py _check_provenance` now **exits fatally** on any
duplicated `(chip_id, period)` pair, naming the regions involved. Fatal rather than a
warning: a silently doubled score is worse than no result. The `region` provenance column
(`panel.py`) makes the cause diagnosable.

### P1a — CORRECTION: polygon-disjoint is NOT sufficient  `verified 2026-08-07`

An earlier draft of this section said overlapping regions could be avoided by constructing
`sweep = basin.difference(priority)`. **That is wrong.** `DLTile.iter_from_shape` returns
every tile *intersecting* the shape, not only tiles contained by it. Measured on the
La Pampa run:

| | |
|---|---|
| AOI polygon area | 0.157500 deg² |
| union of its 80 tile footprints | 0.205330 deg² — **130.4 % of the AOI** |
| tiles not fully inside the polygon | **32 / 80** |
| tile coverage spilling outside the AOI | 30.4 % of AOI area |

So two regions that share a **border** — with exactly zero polygon overlap — still both
receive every tile straddling that border. Tiles are ~5.76 km; any shared boundary of
length *L* duplicates roughly *L / 5.76 km* tiles, each contributing 529 colliding chip_ids.

**AMW's own basin split is polygon-disjoint and would still hit this.** Verified:
`amazon_1 … amazon_6`, sum of parts 548.0044 deg² vs union 548.0044 deg² — **0.000 %
double-covered as polygons**. Yet their tile sets must share the five internal borders.
(`amazon_6` contains `la_pampa_corridor` 100 %.)

**Why it is harmless upstream and fatal here.** AMW's output is dissolved polygons —
`postprocess.dissolve_patches` buffers, unions, explodes, and aggregates confidence as a
*mean*. Union is idempotent: feeding the same detection twice changes nothing. The panel is
one row per chip per period, so a duplicate row is a duplicate observation and every count,
share, mean and κ inflates.

**The correct rules for multi-AOI work (O4 especially):**

1. One AOI per `--panels_dir`. Always.
2. **Deduplicate at the TILE level, not the polygon level.** Polygon disjointness buys
   nothing. Either assign each tile to exactly one region deterministically (e.g. the
   region containing the tile *centroid*), or pool first and drop duplicate `chip_id`s.
3. Merging panels after the fact is legitimate — see P1b for the exact recipe. Never
   `pd.concat` alone.
4. Check any new AOI against existing ones with `shapely.intersects` before a run — but
   remember a *negative* result does not clear you: adjacent regions still share border
   tiles.
5. `analysis.py` now exits fatally on any duplicated `(chip_id, period)`. That is the
   backstop, not the design.

### P1b — The deduplication recipe  `USE THIS`

**A stable `chip_id` is the feature that makes this easy, not the problem.** Because the
index is region-independent, dedup is an exact key join — no geometry matching, no distance
tolerance, no ambiguity. Verified: `cut_chips(pixels, bbox, chip_size, stride, crs)`
(`core/tile_utils.py:150`) takes **no AOI argument**; `i` is a pure function of the 576×576
array shape and `chip_size`/`stride`, and `bbox` is the tile's own global-grid bounds.
Cross-checked on two independent runs (`m2025-01` vs `q2025Q3`): 42,320 shared chip_ids,
identical `(row, col)`, geometry bounds delta `0`. Chip `c37` of a tile is the same ground
in every run and every region, always.

Had the index been region-dependent, dedup would have required fuzzy geometric matching.
It isn't, so:

```python
import pandas as pd, geopandas as gpd

def merge_panels(paths):
    """Pool panels from any set of AOIs into one clean balanced panel.

    Safe because chip_id is a deterministic function of (tile_key, chip index) and the
    tile grid is global: identical chip_id == identical ground, always.
    """
    df = pd.concat([gpd.read_parquet(p) for p in paths], ignore_index=True)

    # Provenance must agree, or the rows are not comparable in the first place.
    for col in ("clear_threshold", "model_sha256", "chip_size", "stride"):
        assert df[col].nunique() == 1, f"mixed {col}: {sorted(df[col].unique())}"

    # Where the same chip appears twice, keep the better-observed composite.
    # n_clear_mean = scenes entering the median; higher is less composite noise.
    df = (df.sort_values("n_clear_mean", ascending=False)
            .drop_duplicates(subset=["chip_id", "period"], keep="first")
            .reset_index(drop=True))

    # The contract: one row per chip per period.
    assert not df.duplicated(subset=["chip_id", "period"]).any()
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
```

Two properties worth stating explicitly:

- **Duplicates are identical, not merely similar.** Same tile, same window, same date range,
  same model → the fetch is cache-served and the probabilities match bit for bit. `keep`
  order therefore only matters when the two runs used different cache states.
- **The tile geometry is frozen by assumption.** `tilesize=552`, `pad=12`, `resolution=10`,
  `chip_size=48`, `stride=24` are project constants and will not change. `chip_id` is
  therefore a permanent identifier for a patch of ground. (`tile.key` itself encodes
  `552:12:10.0:<zone>:<x>:<y>`, so a change would produce different keys anyway.)

## P3 — Stop storing chip geometry; derive it  `PLANNED` (basin scale only)

Measured on `panel_m2026-01.parquet` (schema v2, 42,320 rows, 3.467 MB compressed):

| column | compressed | share |
|---|---|---|
| `geometry` | 0.871 MB | **25.1 %** — largest single column |
| `chip_id` | 0.245 MB | 7.1 % |
| **derivable subtotal** | **1.116 MB** | **32.2 %** |
| `tile_key` + `row` + `col` | 0.004 MB | **0.1 %** |

**~32 % of the panel is reconstructible from 0.1 % of it.** `chip_id` is literally
`f"{tile_key}:c{i}"`, and the polygon is pure arithmetic on the tile bounds
(`core/tile_utils.py:156-182`):

```python
x_per_pixel = (east - west) / W          # W = H = 576
y_per_pixel = (north - south) / H
nw_x = west  + col * x_per_pixel
nw_y = north - row * y_per_pixel
se_x = west  + (col + chip_size) * x_per_pixel
se_y = north - (row + chip_size) * y_per_pixel
```

### P3a — Is the reconstruction lossless? VERIFIED EXACT, 2026-08-07

The whole proposal rests on being able to get the polygons back. Both links of the chain
were tested on real data (`panel_m2026-01.parquet`, 80 tiles, 42,320 chips):

**Link 1 — key string → tile bounds.** `DLTile.from_key("552:12:10.0:19:-15:-263")`
reconstructs the tile from the key alone. Compared against the bounds implied by the stored
chip geometries, over all 80 tiles:

```
max bounds delta : 1.776e-15 deg  =  1.98e-10 m   (float64 rounding)
```

`from_key` is **pure computation** — no network, no Earth Engine, no Descartes Labs
credentials (verified by running it in a clean container with no DL auth configured). The
key encodes `tilesize:pad:resolution:UTM-zone:x:y`, which fully determines the tile on the
global grid.

**Link 2 — tile bounds + row + col → chip polygon.** Rebuilding all 42,320 chips with the
`cut_chips` arithmetic above and comparing to the stored geometry:

```
max bounds error : 0.000e+00 degrees   —  bit-exact, 0 chips differing
```

**Conclusion: dropping `geometry` loses nothing and costs no re-run.** Recovery needs only
`tile_key`, `row`, `col` (0.1 % of the file) plus `descarteslabs`, which is already pinned
in `requirements-lock_2026-07-17.txt`. No re-fetch, no re-inference, no Earth Engine call.

**Residual risk and its mitigation.** The only exposure is a dependency on
`descarteslabs.geo.DLTile` remaining installable and behaviourally stable. It is a pinned,
offline, pure-geometry call, so this is small — but it is not zero, and the panels are meant
to outlive the environment. Belt and braces: write `panels/tile_bounds.parquet` once per AOI
(`tile_key, west, south, east, north`) — **80 rows now, ~220k at basin scale, a few MB once,
not per period**. That removes the library dependency entirely and makes the panel
self-describing. Do this as part of P3, not instead of it.

**Use `tile.geometry.bounds`, never `tile.bounds`.** A `DLTile` exposes both, and they are
in different coordinate systems:

```
DLTile.from_key("552:12:10.0:19:-15:-263")
  .bounds           (417080.0, -1451880.0, 422840.0, -1446120.0)   UTM metres, EPSG:32619
  .geometry.bounds  (-69.7650, -13.1323, -69.7117, -13.0801)       lon/lat, EPSG:4326
```

`cut_chips` is called with the EPSG:4326 pair (`crs='epsg:4326'` is passed alongside), so
the reconstruction must use `.geometry.bounds`.

**Current code is correct** — audited 2026-08-07, every tile-bounds access in the pipeline
uses `.geometry.bounds`: `inference_engine.py:267,772`, `gee.py:189,285`,
`coverage.py:135,164`, `panel.py:339`. There are **zero** uses of `tile.bounds`.
(`tile_utils.py:52` sets `CenteredTile.bounds` in the projected CRS, mirroring DLTile's
convention; nothing reads it. `sam2_build_cog.py` uses `ds.bounds` on a rasterio dataset —
unrelated.) This is a forward-looking warning for whoever implements P3, not a live defect.

**Why it deserves the warning: the mistake is silent.** Tested — writing UTM metres into an
EPSG:4326 GeoParquet raises no error and emits no warning; the values round-trip intact;
and reprojecting them yields `NaN` area with only a `RuntimeWarning`. Downstream that
surfaces as `n/a` in the report rather than as a crash. (An earlier note in this document
described the symptom as "the wrong hemisphere" — that was imprecise: UTM eastings/northings
are 10⁵–10⁶, so read as degrees they fall far outside the valid ±180/±90 range and are not a
shift but nonsense.)

**Scale threshold, not right-or-wrong.**

| | tiles | chips/period | geometry/period | full panel/period |
|---|---|---|---|---|
| La Pampa (now) | 80 | 42,320 | 0.9 MB | 3.5 MB |
| Amazon basin | 219,800 *(`time_tiles.py` default)* | 116.3 M | **~2.4 GB** | **~9.5 GB** |

Basin-monthly: ~114 GB/year of panel, of which ~37 GB/year is derivable. Dropping
`geometry` and `chip_id` cuts each period from ~9.5 GB to ~6.5 GB.

- **Do NOT do this now.** At La Pampa scale the saving is 0.9 MB × 16 = 14 MB/year, and you
  lose the ability to open a panel directly in QGIS — which is worth more than 14 MB.
- **Do it before basin scale**, together with M7 (streaming writes); both are the same
  refactor of the panel writer.
- Implementation: keep `tile_key`, `row`, `col`; write `panels/tile_bounds.parquet` once per
  AOI; add a `rebuild_geometry(df, bounds)` helper in `panel.py` that `analysis.py` calls on
  load. `chip_id` can be regenerated as `tile_key + ":c" + index`, but keep it — 7 % is
  cheap for a column every join and the dedup guard depend on.
- Care: `analysis.py` requires a `geometry` column (`dissolved_area_km2`, `geo_iou`), so the
  rebuild must happen inside `load_panels`, not be left to the caller.

## P2 — Array axis-order convention: `(H, W, B)` vs `(B, H, W)`  `HAZARD — assert, don't trust`

Two incompatible conventions meet in this pipeline:

```
(H, W, B)  rows, cols, bands   ← numpy/TensorFlow: what the MODEL and all fork code want
(B, H, W)  bands, rows, cols   ← rasterio/GDAL:    what the CACHE FILES store
```

The flip happens at exactly four places, all symmetric and all correct today:

| site | direction |
|---|---|
| `core/gee.py:314` (`save_tile`) | `moveaxis(-1, 0)` → `(B,H,W)` for `dst.write` |
| `core/gee.py:353` (`load_tile`) | `moveaxis(0, -1)` → back to `(H,W,B)` |
| `timeseries/coverage.py:130` (`_save`) | `moveaxis(-1, 0)` |
| `timeseries/coverage.py:146` (`_load`) | `moveaxis(0, -1)` |

Everything downstream assumes `(H, W, B)`: `ensure_tile_shape` unpacks
`raster_height, raster_width, _ = raster.shape`; `panel.fill_zero_mismatch` uses
`np.all(pixels == 0, axis=-1)` and `cov[..., 0]`; `chip_coverage_stats` slices
`cov[row:row+cs, col:col+cs]`; `tile_utils.cut_chips` cuts on the first two axes.

**Why this is a hazard rather than a nuisance: a transposition fails silently.**
`ensure_tile_shape` accepts any 3-D array. Hand it a transposed `(13, 576, 576)` and it
reads `height=13, width=576`, then *reflect-pads* it to 576×576 and returns a plausible
array. No exception, no warning — just meaningless probabilities. For coverage it is worse:
`(2, 576, 576)` becomes `height=2`. Both survive to the parquet.

**Recommendation: assert at the boundaries.** The check is O(1) and would convert a silent
scientific error into an immediate crash. Implemented for the fetch choke point in
`panel.py` (2026-08-07); `core/` cannot be touched, so `gee.load_tile` and `save_tile`
remain unguarded — a corrupted or externally-written cache file is the residual risk.

Rule for any new code touching rasters: **name the convention in the docstring, and assert
`arr.shape[-1] == n_bands` immediately after every read.**

## P4 — `EE_PROJECT` silently falls back to upstream's project  `HAZARD — export explicitly`

Three independent, silent ways a run can bill/quota the **wrong** Earth Engine project.

**1. The default is upstream's project.** `gee.py:20`:

```python
EE_PROJECT = os.environ.get("EE_PROJECT", "earthindex")
```

`earthindex` is *upstream's* GCP project, not ours. Also note this executes **once, at
import time** — exporting the variable after `import gee` has no effect.

**2. `.env` is not read by Python.** Nothing in `core/`, `timeseries/` or `time_tiles.py`
loads it. Only `run_staged.sh:112-116` extracts it, and deliberately with `sed` rather than
`source`, because `.env` contains `EE_PROJECT =value` (spaces around `=`) and is therefore
not valid shell — `source` fails with `EE_PROJECT: command not found`. Any direct
invocation bypasses this, **including the smoke test documented in `timeseries/README.md`**:

```bash
python timeseries/run_trials.py --months 2025-07 --no-quarterly --limit_tiles 2
```

**3. An empty export defeats the default.** `os.environ.get(k, default)` returns `default`
only when the key is **absent**. Verified:

```
ABSENT        -> 'earthindex'
SET-BUT-EMPTY -> ''            <- export EE_PROJECT="" gives ee.Initialize(project="")
SET           -> 'my-project'
```

**Consequence.** Quota consumed against the wrong project, or an auth failure whose message
never mentions the project — costing debugging time at exactly the wrong moment (start of a
long run).

**Guard (fork-side, no `core/` edit).** Add to `timeseries/run_trials.py`, above the
`resource_limits` import so it fails before TensorFlow loads:

```python
if not os.environ.get("EE_PROJECT", "").strip():
    sys.exit("FATAL: EE_PROJECT unset or empty. gee.py:20 would fall back to upstream's "
             "'earthindex'. Export it, or launch via timeseries/run_staged.sh.")
```

`run_staged.sh:117` already does the equivalent (`: "${EE_PROJECT:?...}"`); this extends the
same protection to direct invocations.

## P5 — `float32 -> uint16` cache round-trip may not be lossless  `UNVERIFIED — 10-minute check`

**Mechanism.** `get_tile_data` holds pixels as `float32` in memory (`gee.py:213`) but
`save_tile` writes S2 imagery as **`uint16`** (`gee.py:307-308`; confirmed on the live cache:
`dtype: uint16`). And `numpy.astype` **truncates toward zero, it does not round**:

```
in memory (float32)   ->   cached (uint16)   ->   reloaded (float32)
      400.5                     400                    400.0
      400.9                     400                    400.0
```

**Why a non-integral value can arise.** The composite is a per-pixel *median* over the clear
scenes (`gee.py:133`). A median over an **even** number of values is conventionally the mean
of the middle two, which lands on `.5` whenever those two differ by an odd amount.

**Why it matters.** If it occurs, then for the same tile and window:

* a **cold fetch** scores chips from the in-memory `float32` values, but
* a **resumed / cache-hit run** scores them from the truncated `uint16` values,

so *the same chip can get two different probabilities depending on whether the cache was
warm*. That silently breaks the determinism the whole resume-safe design assumes, and it
would be invisible in every output we currently write.

**Scope.** Imagery only. `coverage.py:130` also casts to `uint16`, but `N_CLEAR` / `N_TOTAL`
are integer scene counts by construction, so that cast is exact.

**The check** — insert immediately after `computePixels` in a scratch copy (do NOT edit
`core/`), run one cold tile:

```python
print("non-integral:", np.any(pixels != np.floor(pixels)),
      "  dtype from EE:", pixels.dtype)
```

* `False` -> round-trip is exact. Record it here as VERIFIED and move on.
* `True`  -> pick one: (a) store the fork's own cache as `float32` (≈2× disk: 3.9 -> 7.9 MB
  per tile), (b) round instead of truncate before writing, or (c) accept it and document that
  probabilities are reproducible only to `uint16` precision.

**Do not skip on the grounds that the error is small.** The size of the error is not the
issue — the issue is that it makes results depend on cache state, which is exactly the
property `run_staged.sh`'s resume logic relies on.

## P6 — All-zero imagery tile passed as "fully observed"  `FOUND IN RUN 1 — guard before run 2 analysis`

**Observed, not hypothetical.** Scanning `fill0_mismatch` across all 1,280 tile-periods of
run 1: 1,279 are exactly `0.0`, and **one is `1.0`** (total disagreement):

```
period    m2025-07
tile_key  552:12:10.0:19:-14:-263
status    OK                    <- the pipeline reported success
tile_clear_frac       1.0       <- coverage says every pixel was seen
tile_n_clear_mean     6.07      <- by ~6 clear scenes
fill0_mismatch        1.0       <- but the two validity masks agree NOWHERE
```

Verified against the cached rasters:

```
IMAGERY  S2L1C_552:12:10.0:19:-14:-263_2025-07-01_2025-08-01.tif
         shape (13,576,576)  min=0  max=0   frac_zero=1.0000   <- ENTIRELY BLACK
COVERAGE S2COV_...same key...
         shape (2,576,576)   min=5  max=10  frac_zero=0.0000   <- 5-10 clear scenes
```

And in the panel:

```
529 rows for this tile
prob        max = mean = 6.44618e-05   (IDENTICAL across all 529 chips)
clear_frac  min = mean = 1.0
```

**What happened.** The imagery request returned an all-zero tile even though 5-10 clear
Sentinel-2 scenes covered it. `get_tile_data` raised nothing, so `status=OK` was recorded.
The identical probability across all 529 chips is the signature: every chip was the same
all-zero array.

**Why it matters more than 1-in-1280 suggests.** These 529 chips carry `clear_frac = 1.0`,
so they **pass** `analysis.py:271`'s `obs = g[g.clear_frac >= TAU_OBSERVED]` filter. They are
counted as *confidently observed absence of mining* — contaminating `area_observed_km2`,
which is precisely the number the cloud flag exists to make trustworthy. The coverage raster
detected the problem; nothing acts on it.

**Cause: unknown.** Could be a transient Earth Engine failure returning zeros instead of an
error. Diagnose by deleting that one cache file and refetching; if it reproduces, it is
deterministic and worth a bug report upstream.

**The guard is cheap — `fill0_mismatch` is already computed and stored per tile
(`panel.py:181`), and simply unused by the analysis.** Options, in order:

1. **In `analysis.py`**, join `tile_stats_*.csv` on `tile_key`+`period` and drop (or set
   `clear_frac = 0` for) any tile with `fill0_mismatch` above a small tolerance. Purely
   analysis-side, no re-run needed.
2. **In `panel.py`**, refuse to write rows for a tile whose imagery is entirely zero while
   `N_CLEAR > 0` — record `status="ZERO_IMAGE"` instead of `OK`, so resume logic refetches.
3. **Cheapest sanity check anywhere**: a tile whose 529 chip probabilities are all *bit-
   identical* is degenerate by construction. One line, catches this class of bug generally.

**Apply before analysing run 2.** Run 1's July monthly figure is contaminated by 529
false-observed zeros (~1.25 % of that period's chips) until option 1 is in place.

## P7 — Cache storage is cheap; check Earth Genome's existing cache first  `ACTION: list gs://amw-dev/`

### P7a — There is no cache lifecycle mechanism at all

Searched `core/`, `timeseries/`, `scripts/` for `unlink`, `rmtree`, `os.remove`, eviction, TTL
or size cap. **One hit**, and it is deliberate: `time_tiles.py:83` (`cf.unlink()`, to force a
cold fetch for benchmarking). `--force` rebuilds *panels*, not cache.

The cache therefore **grows monotonically forever**. Current: 6.0 GB (run 1, 16 periods ×
80 tiles). Scaling at 3.94 MB imagery + 0.034 MB coverage per tile-period:

| scope | tiles | periods | cache |
|---|---|---|---|
| La Pampa (run 1) | 80 | 16 | 5.1 GB |
| Madre de Dios full AOI | ~400 | 16 | ~25 GB |
| Amazon basin, monthly | ~197,000 | 12 | **~9.4 TB / yr** |

**This is a feature, not a leak.** Sentinel-2's archive is periodically reprocessed, so a
cached tile is a frozen record of what the model actually saw. Deleting and refetching later
may not reproduce it. Treat the cache as **data**, not scratch.

### P7b — Cost: not a constraint at any plausible scale

Rates checked 2026-08 ([GCS](https://cloud.google.com/storage/pricing),
[S3](https://aws.amazon.com/s3/pricing/)). Six months of retention:

| class | $/GB-mo | 9.4 TB | 20 TB |
|---|---|---|---|
| S3 Standard | 0.023 | $1,297 | $2,760 |
| GCS Standard (dual-region) | 0.022 | $1,241 | $2,640 |
| GCS Coldline | 0.0088 | $496 | $1,056 |
| GCS Archive | 0.0028 | $158 | $336 |
| S3 Glacier Deep Archive | 0.00099 | $56 | $119 |

**Storage is not the cost driver. Two other things are, and they invert the choice:**

1. **Egress.** Reading data *out* of a cloud typically costs far more than storing it. If the
   inference VM sits in the same region/provider as the bucket, egress is free or trivial;
   if it does not, egress will dominate everything in the table above. **Co-locate compute
   with storage** — that decision matters more than the storage class.
2. **Access pattern.** This cache is write-once, **read-many** (every re-analysis re-reads
   it). Cold classes charge per-GB retrieval *and* impose minimum storage durations
   (Nearline 30 d, Coldline 90 d, Archive 365 d — verify current terms). **Archive is the
   wrong class for an active research cache** despite the headline price. Standard or
   Nearline.
3. **Object count.** ~197,000 tiles × 12 periods × 2 files ≈ **4.7 M objects/yr**. Per-request
   (Class A/B) operation charges and listing latency start to matter at that count; O1
   (merge coverage into the imagery request) halves it.

**Conclusion: budget ~$1–3 k/yr for basin-scale cache retention and stop optimising for
bytes.** Compute (M3-revised: inference is 80 % of wall clock) and GEE request quota remain
the binding constraints.

### P7c — Earth Genome already has a cache. Check it before refetching anything.

`core/README.md:59`, verbatim:

> *Earth Genome: cached imagery for 2018 through 2026Q2 can be pulled from `gs://amw-dev/`.*

and `core/README.md:70` shows the local cache directory they use:

```
--image_cache_dir /mnt/tempdisk/amw_image_cache2025_552-12/
                                                     ^^^^^^  tilesize 552, pad 12
```

**Same tile geometry as ours.** So their tiles are drop-in compatible with our cache
directory — the filenames would match ours exactly, *if the date windows match*.

**The open question is the date windows, and it decides everything.** Our cache key is
`{collection}_{tile.key}_{start}_{end}.tif` (`gee.py:178`), so a tile cached for
`2025-01-01_2025-12-31` (annual, as in their documented run) **cannot** serve a
`2025-01-01_2025-02-01` monthly request. Different filename, clean miss.

* If `gs://amw-dev/` holds **annual** composites → useless for the monthly panel, useful only
  for reproducing upstream's published annual products.
* If it holds **quarterly** composites (the "2026Q2" phrasing hints at quarterly binning, but
  may just denote temporal extent) → **directly usable for our quarterly panels**, removing
  every quarterly GEE fetch at basin scale.

**The check is two minutes and decisive** — the filenames state the windows:

```bash
gsutil ls "gs://amw-dev/**/S2L1C_552:12:10.0:19:*" | head -50
```

Read the trailing `_{start}_{end}` on any filename. Do this **before** planning any
basin-scale fetch budget; it could remove a whole cadence's worth of requests. If access is
denied, ask Earth Genome directly — it is their bucket and this fork is a collaboration.

## Order of work

1. O1 + O2 (small, exact, halve-or-better request count) — before Madre de Dios run.
2. O3 experiment during Madre de Dios run planning.
3. O4 design decision alongside cadence choice (register A5), before basin.
4. O5 only as fallback. Re-cost everything from `tile_stats_*.csv` after each stage (register J4).


Check the connection -> storage and caching with the cloud computing 
What should we start with
Panel structure
Vacation
Optimisations - SAM2
