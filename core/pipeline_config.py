"""What a human sets before running the pipeline.

Every value here is shared across stages and across modules, which is why it is
not in a `Config` dataclass: those belong to a single module, and these are the
contract *between* modules.

Two kinds of parameter, and only the first kind lives here:

**Contract parameters** appear in filenames. Every stage finds its input by
constructing a path that embeds them, so all stages must agree -- a threshold is
schema, not a tuning knob. Changing one here changes every derived path at once,
which is the point; changing one on an emitted command line does not, and the
next stage then looks for a file that was never written.

**Behaviour parameters** change what a stage computes but not where anything
lands: `pad`, `tilesize` and `clear_threshold` in `core/gee.py::DataConfig`,
`prior_sigma` and `smoothing_sigma` in `MaskConfig`. Edit the dataclass default.
Note that the outputs will be named identically to any produced before the
change, and nothing in the published provenance records it.

To explore a parameter rather than change the product, bypass the pipeline: call
the underlying script with `--outdir` pointing somewhere separate.
"""
# >> FILE OVERVIEW
# >> Shared constants every pipeline stage must agree on: model name, list of periods,
# >> subregions, thresholds (which appear in filenames) and output folders.
# >> Stages on different machines coordinate ONLY through filenames built from these
# >> values plus files on disk (no scheduler, no database).
# >> Upstream file (Amazon Mining Watch), byte-identical. Read by scripts/pipeline.py.
from __future__ import annotations

from pathlib import Path
from typing import List

# >> Repo root = the folder above core/ (this file is core/pipeline_config.py).
REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# the product
# --------------------------------------------------------------------------

#: Detection model. Paths derive from it, so this is the only place it appears.
# >> Model file stem: models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 = 48 px input,
# >> 6-CNN ensemble, output (N, 6) averaged into one 'confidence' per chip.
MODEL = "48px_v4.10b-18d-20g-21a-22bc-ensemble"

#: Every period the product covers. **Add a period here before running it.**
#: `persist-detections`, `persist-masks` and `stage` read this rather than
#: `--periods`, because they recompute from the whole history; a period absent
#: from the list is invisible to them, so `pipeline.py` refuses one.
# >> Every period tag the product covers: 8 years + 6 quarters (Q125 = Q1 2025).
# >> Downstream persistence steps read this list, not --periods.
ALL_CURRENT_PERIODS: List[str] = [
    "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    "Q125", "Q225", "Q325", "Q425",
    "Q126", "Q226",
]

#: The basin is inferred as six subregions, split at lon -66/-56 and lat -5.
#: Whole-basin runs proved unreliable, so this is the normal path and every
#: full-basin pass concatenates. Names follow
#: data/boundaries/Amazon_ACA/Amazon_ACA_{n}.geojson.
# >> Amazon basin split into 6 pieces (Amazon_ACA_1..6.geojson); pipeline.py emits
# >> one inference command per subregion (+ 1 Andes supplemental) per period.
SUBREGIONS = [1, 2, 3, 4, 5, 6]

# --------------------------------------------------------------------------
# thresholds: the filename contract
# --------------------------------------------------------------------------

#: Raw confidence floor for inference, basin and Andes supplemental.
# >> Raw inference floor: keep chips with confidence > 0.4 (basin).
RAW_THRESHOLD = 0.4
# >> Andes supplemental run uses a lower floor of 0.2.
ANDES_THRESHOLD = 0.2

#: The same two as they appear in filenames. Not derived: the archive writes the
#: basin at two decimals and the supplemental at one, and the files are named
#: that way on the buckets. Formatting cannot reconcile them.
# >> Thresholds as spelled in filenames ('0.40' vs '0.2'); must match files on disk.
RAW_TAG = "0.40"
ANDES_TAG = "0.2"

#: The supplemental has a second spelling: inference_pipeline.py writes two
#: decimals, and `filter` renames to ANDES_TAG on the way into the archive. Both
#: are named here because the raw one is only ever a lookup, and constructing it
#: from a threshold is how it last went wrong.
# >> Andes raw files are written with two decimals ('0.20') before renaming.
ANDES_RAW_TAG = "0.20"

#: Dual-threshold postprocess: (t_main, t_iso). The loose set is the recommended
#: single-period product and what SAM2 is prompted from; the stringent set stands
#: in for corroboration at the provisional edge.
# >> LOOSE: keep a chip if confidence >= 0.43, or >= 0.75 if it is 'isolated'.
# >> This is the recommended product and the set SAM2 is prompted from.
LOOSE = (0.43, 0.75)
# >> STRINGENT: 0.55 main / 0.8 isolated; used for the provisional layer. for newer detections that cannot be double checked against the follwoing year
STRINGENT = (0.55, 0.8)

#: Isolation rule shared by both: k-th nearest neighbour within this distance.
# >> Isolated = the 5th nearest neighbouring chip is farther than 3 km away
# >> (distances measured in one UTM CRS chosen by estimate_utm_crs).
NEIGHBOURS = 5
ISOLATION_KM = 3.0


# >> Build the folder-name fragment for a postprocess setting, e.g.
# >> postprocess_tag(0.43, 0.75) -> 't0.43_d5_3km_t-iso0.75'. ':g' drops trailing 0s.
# >> Used by scripts/pipeline.py (LOOSE_TAG / STRINGENT_TAG) and persistence paths.
def postprocess_tag(t_main: float, t_iso: float,
                    k: int = NEIGHBOURS, isolation_km: float = ISOLATION_KM) -> str:
    """The `t0.43_d5_3km_t-iso0.75` fragment naming a postprocessed product.

    Derived rather than written out, so the directory name and the parameters
    that produced it cannot disagree. Matches `persistence.detection_path`.
    """
    return f"t{t_main:g}_d{k}_{isolation_km:g}km_t-iso{t_iso:g}"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

# >> Folder layout. BASE = data/outputs/<MODEL>/ holds detection GeoJSONs.
CORE = REPO / "core"
SCRIPTS = REPO / "scripts"
BASE = REPO / "data/outputs" / MODEL
# >> SAM2 mask/logit tiles and mosaics.
SAM2 = REPO / "data/outputs/sam2"
# >> GS / SOURCE_COOP: publishing for the AMW website/data release; not part of our
# >> research.
GS = REPO / "data/staging_gs"
SOURCE_COOP = REPO / "data/staging_source-coop"
