# AMW Mining Detector — Research Fork

Trimmed fork of [earthrise-media/mining-detector](https://github.com/earthrise-media/mining-detector) (Amazon Mining Watch).

**Goal:** build a more granular time series of artisanal mining across the Amazon basin. Phase 1: validate the pipeline on a very small test region, then scale up.

Everything not needed for inference/scoring (model training code, website data prep, old 2021–2024 models, published outputs, deprecated code) has been moved to `_archive/` (gitignored, structure mirrors the original repo — restore by moving files back). The upstream README is at `_archive/README_upstream.md`.

## Layout

```
gee/                    Core pipeline (run scripts from this folder)
  gee.py                  Data extraction from GEE + inference engine + configs
  tile_utils.py           Tiling of a region into 576px Sentinel-2 tiles
  inference_pipeline.py   CLI for large-scale inference
  inference.ipynb         Notebook to run the model on a small test area  ← start here
  postprocess.py          Threshold + aggregate patch detections to polygons
  sam2_mask.py            SAM2 segmentation of mine scars (area estimation)
  sam2_build_cog.py       Build cloud-optimized GeoTIFFs from SAM2 masks
  sam2_combine_masks.py   Merge SAM2 mask rasters
  cloud_mask_filter.ipynb Reference: how cloud masking behaves (imports archived
                          viz_tools; read-only reference for the cloud-flag work)
models/
  48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5   Current (Nov 2025) ensemble classifier
  SSL4EO/pretrained/      SSL4EO ViT-S/16 backbone (see setup step 3)
data/
  boundaries/             Region geojsons incl. test_region_small.geojson etc.
  outputs/                Inference output goes here; kept AMW 2018–2025 cumulative
                          baselines in 48px_v0.X-SSL4EO-MLPensemble/cumulative/
scripts/
  wrap_ssl4eo.py          Convert raw SSL4EO .pth checkpoint → .pt the pipeline loads
  concatenate.py          Merge geojson outputs
  dissolve.py             Dissolve overlapping detection polygons
utils/
  gee_data_pull.py        Pull raw Sentinel-2 chips for a region (debugging/cloud work)
```

## Setup

1. GEE auth (needs a Google Cloud project with Earth Engine enabled):
   ```bash
   gcloud auth login
   gcloud auth application-default login
   export EE_PROJECT=YOUR_PROJECT_ID
   ```
2. Python env (already created; deps pinned in `requirements-lock_2026-07-17.txt`):
   ```bash
   source venv/bin/activate
   ```
3. **One-time:** the downloaded backbone at `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pth` is the raw ~720MB SSL4EO checkpoint. `gee.py` expects the wrapped `.pt`. Convert once:
   ```bash
   python scripts/wrap_ssl4eo.py models/SSL4EO/pretrained/dino_vit_small_patch16_224.pth
   ```

## Phase 1: small-area test run

```bash
cd gee
python inference_pipeline.py \
    --model ../models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5 \
    --region_path ../data/boundaries/test_region_small.geojson \
    --pred_threshold 0.85 \
    --start_date 2025-07-01 \
    --end_date 2025-09-30
```

Output geojson lands in `data/outputs/`. For interactive experimentation use `gee/inference.ipynb`. Granularity levers: any date window works (`start_date`/`end_date`), so monthly or even per-composite scoring is a matter of looping windows; any AOI geojson works as `--region_path`.

Then post-process:

```bash
python postprocess.py <output.geojson> --threshold 0.925 \
    --low_area_conf_threshold 0.975 --outpath <post.geojson>
```

Area estimation via SAM2 needs a one-time extra setup (see `gee/README.md`, "Post-processing and masking"), then `sam2_mask.py`, or set `run_sam2=True` in `InferenceConfig` to run it inline.

## Planned modifications (our research)

- **Cloud flag:** per-patch cloud/clear metadata so low-quality composites are flagged rather than silently scored. Relevant code: `DataConfig.clear_threshold` and the Cloud Score+ masking inside `gee/gee.py`; background in `gee/cloud_mask_filter.ipynb`.
- **Area + change:** compute mined area per period and per-period change from our own runs. Building blocks: `postprocess.py`, `scripts/dissolve.py`, SAM2 masks; AMW's cumulative baselines in `data/outputs/48px_v0.X-SSL4EO-MLPensemble/cumulative/` for cross-checking.

## Notes

- Older models (28px v9, 44px v2.x, 48px v3.x) and their outputs are in `_archive/` — only the SSL4EO ensemble is current.
- `requirements.txt` still lists training-era pins; the lock file reflects the working env.
