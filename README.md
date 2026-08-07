<!-- PROVENANCE: FORK-MODIFIED — upstream file modified in this fork; upstream original: earthrise-media/mining-detector -->
# AMW Mining Detector — Research Fork

Fork of [earthrise-media/mining-detector](https://github.com/earthrise-media/mining-detector) (Amazon Mining Watch, AMW).

**Project goal:** build a more granular (monthly/quarterly) time series of artisanal gold mining, piloted on the La Pampa corridor in Madre de Dios, Peru, then scale up.

**Model in use:** `models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5` — upstream's CNN ensemble (six member CNNs: v4.10b, 18d, 20g, 21a, 22b, 22c; the member `.h5` files are in `_archive/models/`, NOT in `models/`). Operating-threshold configs ship alongside (`_config-t0.4.txt`, `_config-t0.43.txt`). It runs directly on Sentinel-2 chips (`embedding_strategy='none'`) — **no SSL4EO backbone download is required**. The earlier SSL4EO+MLP path (phase 1) is archived.

**Last upstream sync:** 2026-08-02 — merge `bac9d30` of upstream `bbbcb2d`. Sync procedure: `docs/GIT_PLAYBOOK.md`.

## Layout

```
core/            Upstream pipeline, inference + eval only (training/embedding
                 notebooks archived). Files present are byte-identical to
                 upstream — do not edit
  inference_engine.py     GEE data extraction + inference engine + configs
  inference_pipeline.py   CLI for large-scale inference
  inference.ipynb         Interactive inference on a small area
  postprocess.py          Threshold + aggregate chip detections to polygons
timeseries/      FORK — the time-series pilot (start here; see timeseries/README.md)
time_tiles.py    FORK — per-tile timing benchmark for cost extrapolation
docs/            FORK — CODE_WALKTHROUGH, DECISION_REGISTER, GIT_PLAYBOOK,
                 OPTIMIZATIONS, LECTURE, PROVENANCE (generated)
models/          ONLY the CNN ensemble in use + its two configs — every other
                 upstream model is archived (none are loaded at runtime)
data/            boundaries/ (la_pampa_corridor, madre_de_dios_mining_aoi, …),
                 timeseries/ (panels, detections, analysis), upstream data
scripts/ utils/  Small utilities (upstream; some carry fork annotations)
_archive/        Gitignored. Everything unused by this project, mirroring the
                 original layout. Upstream's README: _archive/README_upstream.md
                 The deletion ledger: _archive/ARCHIVED_PATHS.txt (generated)
```

## Provenance convention

Every fork-created or fork-modified code/doc file carries a first-line tag:

```
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); …
# PROVENANCE: FORK-MODIFIED — upstream file modified in this fork; …
```

Untagged files are upstream's, kept byte-identical. The full per-file manifest
(including untaggable data files) is `docs/PROVENANCE.md` — regenerate after
every upstream sync with `python scripts/provenance_report.py`.

## Setup

1. Build the environment — see `docs/RUNBOOK.md` §2. Short version: the venv lives at
   `~/venvs/amw` (**not** in the repo), Python **3.11 only**, then
   `python -m pip install -r requirements-lock_2026-07-17.txt`. Activate with
   `source ~/venvs/amw/bin/activate`.
2. GEE auth (Google Cloud project with Earth Engine enabled):
   ```bash
   gcloud auth login && gcloud auth application-default login
   export EE_PROJECT=YOUR_PROJECT_ID
   ```
3. No model download needed — the CNN ensemble is tracked in the repo.

## Run

- **Time-series pilot** (smoke test → overnight run): `timeseries/README.md`
- **Timing benchmark:** `python time_tiles.py --model models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 --region data/boundaries/test_region.geojson …` (full example in the file header)
- **Upstream-style inference on any AOI:** `core/inference_pipeline.py` (CLI) or `core/inference.ipynb` (interactive)
- **Upstream sync:** `bash scripts/sync_upstream.sh` — auto-resolves archived-file conflicts, leaves real ones for review; afterwards run `python scripts/provenance_report.py` (details: `docs/GIT_PLAYBOOK.md`)
