#!/usr/bin/env python3
"""Convert an SSL4EO-S12 DINO checkpoint into the pickled model gee.py expects.

FILE OVERVIEW
=============
One-time model-preparation utility for the AMW pipeline. The inference code
in gee/gee.py loads a READY-TO-USE, full-module pickle of the SSL4EO-S12
ViT-S/16 backbone via torch.load(..., weights_only=False); this script
builds that pickle from whichever checkpoint format is on hand.

Background: SSL4EO-S12 pretrains a ViT-S/16 on Sentinel-2 imagery with
DINO, a self-distillation method that trains TWO networks — a 'student'
and an EMA 'teacher'. The raw ~720 MB .pth checkpoint therefore holds both
state dicts (plus optimizer state), with keys prefixed 'module.backbone.'
(DataParallel 'module.' wrapper + DINO 'backbone.' wrapper). Convention is
to keep the TEACHER weights for downstream use. The wrapped output keeps
only the feature backbone: in the pipeline it embeds each 48x48-px,
13-band Sentinel-2 chip as a 384-d vector, which the six-head Keras MLP
ensemble (384->64->16->1 sigmoid each) then scores for mining activity.

Handles three input formats:
  1. Full SSL4EO-S12 training checkpoint (dict with 'teacher'/'student',
     keys prefixed 'module.backbone.') -- the ~720 MB .pth from the SSL4EO repo.
  2. A timm-ready plain state_dict (e.g. TorchGeo HuggingFace mirror, ~90 MB).
  3. An already-pickled nn.Module (just re-saves to the expected path).

Output: models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt
  (~90 MB pickled torch.nn.Module: ViT-S/16, 13 input channels, no
   classification head; forward() returns (N, 384) pooled features.)
Run from the repo root, venv active:
    python scripts/wrap_ssl4eo.py models/SSL4EO/pretrained/dino_vit_small_patch16_224.pth
"""
import sys                # argv handling and sys.exit for CLI/usage errors
from pathlib import Path  # output-path handling and parent-dir creation

import timm   # PyTorch image-model zoo; supplies the ViT-S/16 architecture
import torch  # checkpoint load/save and the forward-pass smoke test

# Fixed output location, RELATIVE to the CWD — hence "run from the repo
# root". gee.py expects the wrapped model at exactly this path.
OUT = Path("models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt")

def main(src: str):
    """Wrap the checkpoint at ``src`` into a pickled timm ViT-S/16 module.

    Inputs:
        src (str): path to the source checkpoint (.pth). May be (1) a full
            SSL4EO DINO training checkpoint, (2) a plain or nested
            state_dict, or (3) an already-pickled nn.Module — see the
            module docstring for details.
    Returns: None.
    Side effects: writes the pickled model to OUT
        (models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt, ~90 MB),
        creating parent directories as needed; prints progress; calls
        sys.exit if too many keys fail to load; raises AssertionError if
        the patch-embed shape or the smoke-test output shape is wrong.
    Role in pipeline: one-time setup that produces the frozen 384-d
        embedding backbone gee.py loads for inference.
    """
    # weights_only=False permits unpickling arbitrary objects (needed for
    # format 3, a full nn.Module); map_location='cpu' avoids needing a GPU.
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    # Format 3: the file already holds a complete nn.Module — nothing to
    # convert; fall through to the sanity checks and re-save.
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        print("Input is already a full model; re-saving.")
    else:
        # Format 1: raw SSL4EO DINO training checkpoint holding both nets.
        if any(k in ckpt for k in ("teacher", "student")):
            # Prefer the teacher (EMA of the student): the standard choice
            # for downstream transfer from DINO-style self-distillation.
            which = "teacher" if "teacher" in ckpt else "student"
            print(f"Full DINO checkpoint detected; using '{which}' weights.")
            raw_sd = ckpt[which]  # state_dict of the chosen network
            sd = {}  # cleaned state_dict with timm-compatible key names
            for k, v in raw_sd.items():
                # Strip the DataParallel wrapper prefix ('module.').
                if k.startswith("module."):
                    k = k[len("module."):]
                # Keep only backbone tensors, dropping the 'backbone.'
                # prefix; DINO projection-head keys are discarded here.
                if k.startswith("backbone."):
                    sd[k[len("backbone."):]] = v
            print(f"Extracted {len(sd)} backbone tensors "
                  f"(from {len(raw_sd)} total in '{which}').")
        # Format 2a: dict that nests the weights under 'state_dict'.
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt  # assume plain state_dict (format 2b, timm-ready keys)

        # Build the target architecture: ViT-S/16 (384-d embedding dim).
        # in_chans=13 rebuilds the patch-embed conv for the 13 Sentinel-2
        # bands (instead of RGB's 3); num_classes=0 removes the classifier
        # head so forward() returns pooled 384-d features, not logits.
        model = timm.create_model("vit_small_patch16_224", in_chans=13,
                                  num_classes=0)
        # strict=False tolerates mismatched keys; instead of trusting it
        # blindly, the mismatch report is inspected below.
        result = model.load_state_dict(sd, strict=False)
        print(f"Missing keys: {result.missing_keys}")  # model params sd lacked
        print(f"Unexpected keys: {result.unexpected_keys}")  # sd entries with no slot
        # Number of tensors actually copied into the model.
        n_loaded = len(sd) - len(result.unexpected_keys)
        print(f"Loaded {n_loaded} weight tensors.")
        # A couple of missing keys can be benign; more than 2 means the
        # checkpoint layout does not match ViT-S/16 — abort loudly.
        if len(result.missing_keys) > 2:
            sys.exit("ERROR: too many missing keys -- wrong checkpoint format?")

    # Sanity check: the patch-embed conv weight must be shaped
    # (384 embed dims, 13 input bands, 16x16 patch). Guards against having
    # accidentally built/loaded a 3-channel RGB model.
    w = dict(model.named_parameters())["patch_embed.proj.weight"]
    assert tuple(w.shape) == (384, 13, 16, 16), f"Bad patch embed: {w.shape}"

    model.eval()  # inference mode (freezes dropout etc.) before pickling
    # Smoke test: forward one zero image (batch 1, 13 bands, 224x224 px)
    # and confirm the output is a single 384-d pooled feature vector.
    with torch.no_grad():
        out = model(torch.zeros(1, 13, 224, 224))
    assert tuple(out.shape) == (1, 384), f"Bad output shape: {out.shape}"
    print(f"Smoke test OK: input (1,13,224,224) -> output {tuple(out.shape)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)  # ensure output dir exists
    # Save the WHOLE module (architecture + weights) as a pickle — not just
    # a state_dict — because gee.py loads it with weights_only=False and
    # uses it directly without rebuilding the architecture in code.
    torch.save(model, OUT)
    print(f"Saved ready-to-use model to {OUT} "
          f"({OUT.stat().st_size/1e6:.0f} MB)")  # report final size (~90 MB)

if __name__ == "__main__":
    # Exactly one CLI argument (source checkpoint path) is required;
    # otherwise print the module docstring as usage help and exit non-zero.
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])  # run the conversion
