#!/usr/bin/env python3
"""
Deriving mine-scar masks from saved SAM2 logits.

READ THIS BEFORE THRESHOLDING A ``-logits.tif``.

The saved logits **are** the field the production mask thresholds::

    stored = smooth(clip(upsampled_log_odds, ±16))
    mask   = stored > 0

So thresholding is the whole operation, and a stricter cutoff is a true
re-threshold::

    from sam2_logits import mask_from_logits
    mask = mask_from_logits(logits_array)                   # the product mask
    mask = mask_from_logits(logits_array, threshold=1.5)    # a stricter t_prov

Thresholding commutes with the max-reduce that merges overlapping tiles --
``max(a,b) > t`` is identically ``(a>t) or (b>t)`` -- so a logits *mosaic* may be
thresholded directly.

The ±16 clamp is lossless with respect to the mask: verified bit-identical to
the unclamped mask on every real tile tested. See ``MaskConfig.logit_clamp``.

``smoothing_sigma`` is fixed at 2.5 and baked in. Changing it now requires
re-running SAM2, which is deliberate: measured over 750 tiles, sigma across 0-5
moves basin mask area only ~2%, and an IoU scan against hand annotations found
the optimum at 1-1.5 worth +0.015 mean IoU while *worsening* the area ratio. The
two criteria conflict and both effects are small, so the flexibility did not pay
for a stored field that silently mis-thresholds.

Background and measurements: docs/design/persistence-planning.md.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.ndimage as ndi

# >> FILE OVERVIEW: small shared helpers for SAM2 logit rasters (upstream AMW file).
# >> "logit" = log-odds score; > 0 means SAM2 thinks the pixel is more likely mine than
# >> not. Holds the single source of truth for smoothing sigma, clamp, threshold and mask
# >> nodata, imported by inference_engine.MaskConfig, scripts/write_mask_config.py and
# >> scripts/convert_logits_to_smoothed.py; sam2_persistence.py imports MASK_NODATA.
# >> Gotcha: the docstring above says stored = smooth(clip(...)); new runs actually store
# >> clip(smooth(...)) (inference_engine, "Clamp AFTER smoothing"). Only converted legacy
# >> tiles are clip-first.
#: Gaussian regularization applied to upsampled logits before thresholding.
#: Single source of truth; ``MaskConfig.smoothing_sigma`` defaults to this.
# >> Gaussian blur width in PIXELS of the upsampled ~10 m mask grid (2.5 px ~ 25 m).
DEFAULT_SMOOTHING_SIGMA = 2.5

#: Saturation limit (log-odds) for persisted logits. Lossless w.r.t. the mask.
# >> Logits are capped to +/-16 log-odds (probability 1 - 1e-7): shrinks files, same mask.
DEFAULT_LOGIT_CLAMP = 16.0

#: Log-odds cutoff for the production mask (probability 0.5).
# >> 0 log-odds = probability 0.5; the product mask is "logit > 0".
PRODUCTION_THRESHOLD = 0.0

#: uint8 sentinel for "not observed" in mask rasters.
# >> Mask rasters use 0 = no mine, 1 = mine, 2 = not observed (nodata).
MASK_NODATA = 2


# >> Blurs a logit raster, ignoring NaN (unobserved) pixels instead of spreading them.
# >> Input: 2-D array (H, W), any float dtype, NaN = nodata. Output: float64 (H, W), NaN
# >> kept. Uses scipy.ndimage.gaussian_filter. Method = "normalised convolution": blur
# >> values with NaN set to 0, blur the 0/1 validity map, divide. Used by mask_from_logits
# >> (optional replay), write_mask_config.py and convert_logits_to_smoothed.py (to fix
# >> legacy files saved without smoothing).
def smooth_logits(logits: np.ndarray,
                  smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA
                  ) -> np.ndarray:
    """Replay the Gaussian regularization on logits that lack it.

    NaN marks unobserved ground. ``ndi.gaussian_filter`` would smear NaN across
    the whole neighbourhood, so NaNs are held out and the result renormalized
    by the smoothed validity mask -- equivalent to smoothing over observed
    pixels only.
    """
    # >> sigma 0/None means "no smoothing": just return a float64 copy.
    if not smoothing_sigma:
        return np.asarray(logits, dtype=np.float64)

    values = np.asarray(logits, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.all():
        # >> Fast path: no NaN anywhere, so a plain Gaussian blur is exact.
        return ndi.gaussian_filter(values, sigma=smoothing_sigma)

    # >> Numerator: blur of values with NaN replaced by 0.
    filled = np.where(valid, values, 0.0)
    num = ndi.gaussian_filter(filled, sigma=smoothing_sigma)
    # >> Denominator: blur of the validity map = how much real data fed each pixel.
    den = ndi.gaussian_filter(valid.astype(np.float64), sigma=smoothing_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        # >> Divide to renormalise; where no valid data contributed, result is NaN.
        out = np.where(den > 0, num / den, np.nan)
    # >> Put NaN back on originally-unobserved pixels (no filling in of gaps).
    return np.where(valid, out, np.nan)


# >> Turns a logit raster into a yes/no mine mask by comparing against a threshold.
# >> Input: (H, W) logits (NaN = nodata). Output: bool (H, W), or uint8 0/1/2 if as_uint8.
# >> threshold is in log-odds (0 = production; higher = stricter "provisional" mask).
# >> Gotcha: in the bool form NaN > t is False, so "not observed" silently becomes 0;
# >> only as_uint8=True keeps nodata as 2.
def mask_from_logits(logits: np.ndarray,
                     threshold: float = PRODUCTION_THRESHOLD,
                     smoothing_sigma: Optional[float] = None,
                     as_uint8: bool = False) -> np.ndarray:
    """Mask from saved logits: a plain threshold.

    Logits are stored smoothed, so thresholding is the whole operation and
    reproduces the production mask at ``threshold=0``. Raise the threshold for a
    stricter provisional mask (``t_prov,mask``).

    Args:
        logits: saved log-odds for a single tile. NaN = unobserved.
        threshold: log-odds cutoff. 0 is the production operating point.
        smoothing_sigma: replay smoothing before thresholding, for logits that
            lack it. Applying it to a smoothed file smooths twice.
        as_uint8: return 0/1/``MASK_NODATA`` uint8 instead of a boolean array.
            Only this form can represent "not observed".

    Returns:
        Boolean mask, or uint8 with :data:`MASK_NODATA` where input was NaN.
    """
    # >> Optionally replay smoothing first (only for logits stored WITHOUT smoothing).
    field = logits if smoothing_sigma is None else smooth_logits(
        logits, smoothing_sigma)
    # >> The actual thresholding: one comparison per pixel.
    mask = field > threshold

    if not as_uint8:
        return mask

    # >> Convert True/False to 1/0, then mark non-finite (NaN) pixels as 2 = nodata.
    out = mask.astype(np.uint8)
    out[~np.isfinite(field)] = MASK_NODATA
    return out


# >> Merges several same-grid uint8 masks: a pixel is mine if ANY tile says mine,
# >> nodata only if EVERY tile is nodata, otherwise 0. Output uint8 (H, W).
# >> Gotcha: the docstring warns against max-reduce-then-threshold, but the module
# >> docstring (correctly) says max(a,b) > t equals (a>t) or (b>t); the real
# >> difference is only how nodata is handled.
def mosaic_masks(masks) -> np.ndarray:
    """Union-merge per-tile uint8 masks that already share a grid.

    1 wins over 0, and nodata only survives where every input is nodata. This
    is the correct merge for masks; do not substitute a max-reduce over logits
    followed by a single threshold (see the module docstring).
    """
    # >> Stack N masks: list of (H, W) -> array (N, H, W).
    stack = np.stack([np.asarray(m, dtype=np.uint8) for m in masks])
    # >> Boolean (N, H, W): True where that input says "not observed".
    nodata = stack == MASK_NODATA
    # >> Reduce over axis 0 (the N inputs): all-nodata -> 2, any-1 -> 1, else 0.
    return np.where(nodata.all(axis=0), MASK_NODATA,
                    np.where((stack == 1).any(axis=0), 1, 0)).astype(np.uint8)
