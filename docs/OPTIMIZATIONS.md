# Optimization Backlog — Earth Engine request architecture & scaling

Status of scaling optimizations: done, planned, and considered-but-rejected.
Companion to `DECISION_REGISTER.md` (§B, §J). Voice: dense, minimal prose.
Impact figures assume basin scale (~220k tiles) unless noted.

**Status codes** — `DONE` implemented · `PLANNED` do before next scale-up ·
`EXPERIMENT` measure first · `CONTINGENCY` only if quotas force it · `REJECTED`.

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
| M3 | Cross-tile CNN batch inference | REJECTED — inference is not the bottleneck (network wait is) |
| M4 | Fetch per-scene imagery, composite locally (enables any operator) | REJECTED — 6–18× transfer volume; defeats D1 |
| M5 | Reconstruct quarterly imagery from monthly composites | REJECTED — median nonlinearity; methodologically invalid |
| M6 | Drop B10 (13→12 bands) | REJECTED — band set is a training contract (register B2) |

## Order of work

1. O1 + O2 (small, exact, halve-or-better request count) — before Madre de Dios run.
2. O3 experiment during Madre de Dios run planning.
3. O4 design decision alongside cadence choice (register A5), before basin.
4. O5 only as fallback. Re-cost everything from `tile_stats_*.csv` after each stage (register J4).
