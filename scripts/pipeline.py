#!/usr/bin/env python3
"""
Driver for the detection -> mask -> published-product pipeline.

Runs on the VM. Stages are idempotent (skip what exists) and verify their own
output, because the failure modes here are silent: in one week we saw
``gsutil cp -I`` report success while copying 2 of 15,752 files, ``sam2_mask.py``
default to 2023 imagery when dates were omitted, and shell globs fail mid-list at
ARG_MAX. None announced itself; all are caught by counting outputs.

Some stages are **run by a human**, not by this tool -- detection and SAM2
inference take hours across parallel tmux sessions on several VMs. For those the
driver *emits* commands with every parameter derived from the period list, so
dates, cache directories and boundary arguments are never typed. ``publish``
is emitted rather than executed for a different reason: it is outward-facing, and
the rasters want reviewing first.

    python scripts/pipeline.py --list
    python scripts/pipeline.py inference --periods 2026-Q3
    python scripts/pipeline.py postprocess persist-detections
    python scripts/pipeline.py publish

Design and layout: docs/design/pipeline.md
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

# >> FILE OVERVIEW: the conductor script (upstream AMW). It does not do science itself; it
# >> either PRINTS the long commands a human runs in tmux on several VMs (HUMAN stages),
# >> or RUNS the short bookkeeping scripts (concat, filter, postprocess, persistence,
# >> cog). Stage order: review-config -> inference -> concat -> filter -> postprocess ->
# >> persist-detections -> mask-annual -> mask-quarterly -> cog -> persist-masks -> stage
# >> -> manifest -> publish. Machines coordinate only through file names on disk (the
# >> naming contract in core/pipeline_config.py); no scheduler, no database. Gotcha: the
# >> docstring example "--periods 2026-Q3" is rejected by Period.parse, which only accepts
# >> "2026" or quarter tags like "Q326".
REPO = Path(__file__).resolve().parent.parent
# >> Make core/ importable (periods.py, pipeline_config.py).
sys.path.insert(0, str(REPO / "core"))

from periods import Period
from pipeline_config import (ALL_CURRENT_PERIODS, ANDES_RAW_TAG, ANDES_TAG,
                             ANDES_THRESHOLD, BASE, CORE, ISOLATION_KM, LOOSE,
                             MODEL, NEIGHBOURS, RAW_TAG, RAW_THRESHOLD, SAM2,
                             SCRIPTS, STRINGENT, SUBREGIONS, postprocess_tag)

# >> Folder/filename fragments like "t0.43_d5_3km_t-iso0.75": LOOSE = (0.43 main
# >> threshold, 0.75 isolated), STRINGENT = (0.55, 0.8); d5 = 5th nearest neighbour, 3km =
# >> isolation.
LOOSE_TAG = postprocess_tag(*LOOSE)
STRINGENT_TAG = postprocess_tag(*STRINGENT)

# >> HUMAN stages: the driver only PRINTS their commands (hours-long GPU/Earth Engine
# >> jobs, or outward-facing uploads); everything else is executed by the driver itself.
HUMAN = {"review-config", "inference", "mask-annual", "mask-quarterly", "publish"}

#: These recompute from the whole history rather than from the periods named on
#: the command line, so they read ALL_CURRENT_PERIODS. Passing them one period
#: would compute onset with nothing to corroborate against.
# >> WHOLE_HISTORY stages ignore --periods and always use ALL_CURRENT_PERIODS, because
# >> persistence (k=2 detections in a 2-year window) needs every year to confirm onsets.
WHOLE_HISTORY = {"persist-detections", "persist-masks", "stage", "manifest"}

# >> Dependency order; --list prints it.
ORDER = ["review-config", "inference", "concat", "filter", "postprocess", "persist-detections",
         "mask-annual", "mask-quarterly", "cog", "persist-masks", "stage",
         "manifest", "publish"]


# >> Period tag -> [start_date, end_date] strings, e.g. "2024" -> ["2024-01-01",
# >> "2024-12-31"]. Gotcha: these end dates are fed to Earth Engine, where the end is
# >> EXCLUSIVE, so the last day of every period (31 Dec, 31 Mar, ...) is never imaged.
def dates(tag: str):
    return Period.parse(tag).date_span.split("_")


# >> Local image-cache folder for one period on the VM's scratch disk:
# >> /mnt/tempdisk/amw_image_cache<tag>_552-12/ (e.g. ..._cache2024_552-12/,
# >> ..._cacheQ125_...). "552-12" = CNN tilesize 552 px + pad 12 px. /mnt/tempdisk is a
# >> big local (non-backed-up) disk, so re-runs reuse downloaded pixels instead of asking
# >> Earth Engine again. Gotcha: mask-annual/-quarterly reuse this same folder although
# >> SAM2 tiles are 256 px.
def cache_dir(tag: str) -> str:
    """Image cache naming from the operator's runbook, one directory per period.

    The 552-12 suffix is DataConfig's tilesize and pad, which fix the tile
    geometry a cached tile was cut to; a cache is only reusable by a run that
    agrees on both.
    """
    return f"/mnt/tempdisk/amw_image_cache{Period.parse(tag).tag}_552-12/"


# --------------------------------------------------------------------------
# emitted commands (human-run)
# --------------------------------------------------------------------------

# >> STAGE review-config (HUMAN, prints only): a checklist of the settings that end up in
# >> file names (model, periods, subregions, thresholds). Exists because changing a value
# >> on one command line silently breaks the file-name hand-off to the next stage. Returns
# >> a list of "#" lines that main() prints.
def cmds_review_config(periods: Sequence[str]) -> List[str]:
    """Stage 0: what a human sets, and where."""
    annual = [p for p in ALL_CURRENT_PERIODS if Period.parse(p).is_annual]
    quarters = [p for p in ALL_CURRENT_PERIODS if not Period.parse(p).is_annual]
    return [
        "# 0. Review core/pipeline_config.py before running anything.",
        "#",
        f"#    MODEL      {MODEL}",
        f"#    PERIODS    {len(annual)} annual   {' '.join(annual)}",
        f"#               {len(quarters)} quarterly {' '.join(quarters)}",
        f"#    SUBREGIONS {SUBREGIONS}",
        f"#    thresholds raw {RAW_THRESHOLD:g} (basin) / {ANDES_THRESHOLD:g} (andes)",
        f"#               loose     {LOOSE_TAG}",
        f"#               stringent {STRINGENT_TAG}",
        "#",
        "#    Add the period you are about to run, then pass it as --periods.",
        "#    Every --periods value must be a member of the list: "
        "persist-detections,",
        "#    persist-masks and stage read it rather than --periods, because they",
        "#    recompute from the whole history. A missing period is invisible to",
        "#    them and will not reach the product. --all uses the whole list.",
        "#",
        "#    Those values appear in filenames, so every stage must agree on them.",
        "#    Changing one on an emitted command line does not reconfigure the",
        "#    pipeline -- the next stage looks for a file that was never written.",
        "#",
        "#    Parameters that change a computation but not a path -- pad, tilesize,",
        "#    clear_threshold in core/gee.py::DataConfig; prior_sigma,",
        "#    smoothing_sigma in MaskConfig -- are edited in the dataclass. The",
        "#    outputs will be named identically to any produced before the change,",
        "#    and nothing in the published provenance records it, so note it.",
        "#",
        "#    To explore a parameter rather than change the product, call the",
        "#    underlying script directly with --outdir somewhere separate.",
    ]


# >> STAGE inference (HUMAN, prints only): the CNN detection runs, 7 independent commands
# >> per period = 6 Amazon_ACA subregions (threshold 0.4) + Andes supplemental (0.2). Each
# >> downloads Earth Engine composites, scores 48 px chips and writes raw detections
# >> GeoJSON. Split up so they can run in parallel across VMs; outputs are merged later by
# >> concat. Gotcha: 0.4 / 0.2 are typed here instead of using RAW_THRESHOLD /
# >> ANDES_THRESHOLD.
def cmds_inference(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in periods:
        s, e = dates(tag)
        # >> One command per Amazon_ACA subregion (SUBREGIONS = [1..6]).
        for r in SUBREGIONS:
            out.append(
                f"python inference_pipeline.py --model ../models/{MODEL}.h5 "
                f"--region_path ../data/boundaries/Amazon_ACA/Amazon_ACA_{r}.geojson "
                f"--start_date {s} --end_date {e} "
                f"--image_cache_dir {cache_dir(tag)} --pred_threshold 0.4 --tries 3")
        # >> Plus one Andes supplemental run at the lower 0.2 score floor.
        out.append(
            f"python inference_pipeline.py --model ../models/{MODEL}.h5 "
            f"--region_path ../data/boundaries/andes_supplemental.geojson "
            f"--start_date {s} --end_date {e} "
            f"--image_cache_dir {cache_dir(tag)} --pred_threshold 0.2 --tries 3")
    return out


# >> STAGE mask-annual (HUMAN, prints only): SAM2 pixel masks for each annual period, two
# >> runs per year: the LOOSE post-processed basin detections, and the filtered Andes
# >> detections. --cog makes sam2_mask.py build the mosaics itself. Output folders are
# >> named after the detections file (see run_dirs).
def cmds_mask_annual(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in (t for t in periods if Period.parse(t).is_annual):
        s, e = dates(tag)
        pp = (f"../data/outputs/{MODEL}/postprocessed_{LOOSE_TAG}/"
              f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}_{LOOSE_TAG}.geojson")
        andes = (f"../data/outputs/{MODEL}/raw_detections/andes_supplemental/"
                 f"andes_supplemental_{MODEL}_{ANDES_TAG}_{s}_{e}.geojson")
        out.append(f"python sam2_mask.py {pp} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
        out.append(f"python sam2_mask.py {andes} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
    return out


# >> STAGE mask-quarterly (HUMAN, prints only): SAM2 masks for each quarter, prompted only
# >> by the NEW detections of that quarter (patch_diffs/*growth_<tag>.geojson written by
# >> persist-detections), so it must run after persist-detections.
def cmds_mask_quarterly(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in (t for t in periods if not Period.parse(t).is_annual):
        s, e = dates(tag)
        diff = (f"../data/outputs/{MODEL}/cumulative/patch_diffs/"
                f"Amazon_ACA_{MODEL}_growth_{tag}.geojson")
        out.append(f"python sam2_mask.py {diff} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
    return out


# >> STAGE publish (HUMAN, prints only): publishing for the AMW website/data release; not
# >> part of our research.
def cmds_publish(periods: Sequence[str]) -> List[str]:
    """Emitted, never executed: outward-facing, and the rasters want review.

    Checks the staging trees before emitting anything. Whatever is in them gets
    published, and they are assembled fresh each run, so a file stage() did not
    put there is a file that should not go out -- a quarto preview once left
    README.html and twelve bootstrap assets in the public tree.
    """
    # amw-published is the store of record and is versioned; amw-dev/published is
    # its backup, synced bucket-to-bucket so nothing round-trips through a laptop
    # -- every silent transfer failure we have hit was a local<->bucket sync.
    record = "gs://amw-published"
    backup = "gs://amw-dev/published"
    coop = "s3://earthgenome/amazon-mining-watch"
    warn: List[str] = []
    try:
        from stage_outputs import check_trees
        report = check_trees(periods)
    except Exception as exc:                      # never block the emit on this
        warn = [f"# NOTE: could not check the staging trees ({exc})", ""]
        report = {}

    def _lines(kind: str, files: List[str], tree: str, note: str) -> List[str]:
        out = [f"# WARNING: {len(files)} {kind} file(s) in {tree}/. {note}"]
        out += [f"#     {f}" for f in files[:10]]
        if len(files) > 10:
            out.append(f"#     ... and {len(files) - 10} more")
        return out + [""]

    for tree, r in report.items():
        if r["stray"]:
            warn += _lines("unexpected", r["stray"], tree.name,
                           "Staging did not put these here and they WILL be "
                           "published; remove them.")
        if r["missing"]:
            warn += _lines("MISSING", r["missing"], tree.name,
                           "Expected but absent -- an incomplete product would go "
                           "out. Re-run `pipeline.py stage`.")

    return warn + [
        "# review the rasters before running any of this",
        "",
        "# 1. store of record",
        "# add -c if objects were pre-populated by cp: rsync compares mtime, cp",
        "# does not set it, so those objects re-upload however identical they are",
        f"gsutil -m rsync -r data/staging_gs/ {record}/",
        "# Verify by name, not by count. Two different tools have silently",
        "# dropped files on this project: gsutil cp -I reported success having",
        "# copied 2 of 15,752, and aws s3 sync dropped 2 of 48. A count tells",
        "# you something is missing; this tells you which. Empty output = clean.",
        f"diff <(cd data/staging_gs && find . -type f | sed 's|^\./||' | sort) \\",
        f"     <(gsutil ls '{record}/**' | sed 's|^{record}/||' | sort)",
        "",
        "# 2. backup, server-side (no egress, no local round trip)",
        f"gsutil -m rsync -r -d {record}/ {backup}/",
        "",
        "# 3. public subset.",
        "#",
        "#    FIRST: Source Cooperative issues temporary, AWS-compatible",
        "#    credentials -- there is no long-lived key to configure. Log in at",
        "#    https://source.coop, open the earthgenome/amazon-mining-watch",
        "#    repository, and copy the credential block it shows into this shell:",
        "#",
        "#      export AWS_ACCESS_KEY_ID=...",
        "#      export AWS_SECRET_ACCESS_KEY=...",
        "#      export AWS_SESSION_TOKEN=...",
        "#      export AWS_DEFAULT_REGION=\"us-east-1\"",
        "#      export AWS_ENDPOINT_URL=\"https://data.source.coop\"",
        "#",
        "#    All five matter. AWS_ENDPOINT_URL is what points the CLI at Source",
        "#    Cooperative -- without it the commands below address real AWS S3",
        "#    instead, and either fail on credentials or reach a bucket that is",
        "#    not ours.",
        "#",
        "#    The first three expire, and 1.4 GB is long enough to outlive them.",
        "#    If the upload dies with an auth error, re-copy the block and re-run:",
        "#    sync compares size and mtime, so it resumes instead of restarting.",
        "#",
        "#    sync, not cp --recursive, for that reason and because a quarterly",
        "#    refresh changes only a handful of files. No --delete: anything",
        "#    removed from the staging tree stays on the bucket until deleted by",
        "#    hand, which is what keeps archived/ -- present on the bucket, absent",
        "#    from staging -- from being swept away.",
        f"aws s3 sync data/staging_source-coop/ {coop}/",
        "# Same verification. archived/ is on the bucket and not in staging, so",
        "# it is excluded -- otherwise the comparison can never come out clean",
        "# and the check gets ignored. Empty output = clean; re-run the sync for",
        "# anything listed, which is what the 2-of-48 drop needed.",
        f"diff <(cd data/staging_source-coop && find . -type f | sed 's|^\./||' | sort) \\",
        f"     <(aws s3 ls --recursive {coop}/ | awk '{{print $4}}' \\",
        f"        | sed 's|^amazon-mining-watch/||' | grep -v '^archived/' | sort)",
    ]


# >> Stage name -> function that PRINTS commands (HUMAN stages).
EMITTERS = {"review-config": cmds_review_config,
            "inference": cmds_inference, "mask-annual": cmds_mask_annual,
            "mask-quarterly": cmds_mask_quarterly, "publish": cmds_publish}


# --------------------------------------------------------------------------
# executed stages
# --------------------------------------------------------------------------

# >> Runs one command as a child process, printing it first ("$ ..."). With --dry-run it
# >> only prints. Returns the exit code (0 = success); stages OR codes together.
def run(cmd: List[str], dry: bool) -> int:
    # flush before handing the terminal to a child: our stdout is buffered when
    # piped, the child's is not, so without this the stage headers and the "$ cmd"
    # lines land after the output they introduce, and a multi-stage log becomes
    # impossible to attribute.
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


# >> STAGE concat (executed): joins the 6 subregion raw-detection GeoJSONs of each period
# >> into one basin file raw_detections/Amazon_ACA_<model>_0.40_<start>_<end>.geojson via
# >> scripts/concatenate.py. Idempotent: skips periods whose output already exists.
# >> Output-count check: needs exactly 6 of 6 parts (or 0); a partial set is a hard stop,
# >> because a 5-of-6 basin file would look complete to every later stage.
def stage_concat(periods, dry) -> int:
    out = BASE / "raw_detections"
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    missing: List[str] = []
    for tag in periods:
        s, e = dates(tag)
        dest = out / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
        # >> Idempotence: already concatenated -> skip.
        if dest.is_file():
            continue
        # >> "?" = one character, the subregion number 1..6.
        pattern = f"Amazon_ACA_?_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
        parts = sorted(glob.glob(str(BASE / pattern)))
        # >> The count check: 0 parts = not run yet; 6 = complete; anything else = stop.
        if len(parts) not in (0, len(SUBREGIONS)):
            # A partial concatenation is named exactly like a complete basin and
            # every later stage accepts it, so this stops rather than warns.
            raise SystemExit(
                f"{tag}: {len(parts)} of {len(SUBREGIONS)} subregion parts.\n"
                f"  Concatenating these would write a basin file missing whole "
                f"regions,\n  under a name nothing downstream can distinguish "
                f"from a complete one.\n"
                f"  Found: {[Path(x).name for x in parts]}")
        if not parts:
            print(f"    {tag}: no subregion parts matching {pattern}")
            missing.append(tag)
            continue
        # >> |= : remember any failure (non-zero exit) while continuing the loop.
        rc |= run([sys.executable, str(SCRIPTS / "concatenate.py"), *parts,
                   "--outpath", str(dest)], dry)
    if missing:
        print(f"    {len(missing)} period(s) had no inference output: "
              f"{', '.join(missing)}")
        rc |= 1
    return rc


# >> STAGE filter (executed): clips the Andes supplemental detections to the Andes
# >> boundary polygon with scripts/geo_filter.py, writing
# >> raw_detections/andes_supplemental/... Exists because the Andes run's box overlaps the
# >> basin; only detections inside the Andes area should be added. Idempotent: skips
# >> existing outputs. Missing inputs are only printed.
def stage_filter(periods, dry) -> int:
    """Clip andes detections to their boundary, writing straight to their home.

    ``geo_filter.py`` requires ``--outpath``, so the destination is passed rather
    than reconstructed from a ``-filt`` sibling afterwards: the filtered file is
    written once, in the one place both ``persistence.py`` (via
    ``--supplemental_dir``) and ``stage_outputs.raw_andes`` look for it. The
    ``0.20`` -> ``0.2`` shift between the inference output and the filtered name is
    the archive's existing convention, and is now visible in the command instead of
    buried in a rename.
    """
    out = BASE / "raw_detections" / "andes_supplemental"
    out.mkdir(parents=True, exist_ok=True)
    boundary = REPO / "data/boundaries/andes_supplemental.geojson"
    rc = 0
    for tag in periods:
        s, e = dates(tag)
        dest = out / f"andes_supplemental_{MODEL}_{ANDES_TAG}_{s}_{e}.geojson"
        if dest.is_file():
            continue
        src = BASE / f"andes_supplemental_{MODEL}_{ANDES_RAW_TAG}_{s}_{e}.geojson"
        if not src.is_file():
            print(f"    {tag}: no andes inference output at {src}")
            continue
        rc |= run([sys.executable, str(SCRIPTS / "geo_filter.py"),
                   str(src), str(boundary), "--outpath", str(dest)], dry)
    return rc


# >> STAGE postprocess (executed): for both LOOSE and STRINGENT settings, runs
# >> core/postprocess.py on each period's basin raw detections (dedupe, main/isolated dual
# >> threshold, dissolve) and moves the result into postprocessed_<tag>/. Idempotent:
# >> skips outputs that exist. Missing raw input is printed but does not fail the stage.
def stage_postprocess(periods, dry) -> int:
    rc = 0
    for t_main, t_iso in (LOOSE, STRINGENT):
        tag_ = postprocess_tag(t_main, t_iso)
        target = BASE / f"postprocessed_{tag_}"
        target.mkdir(parents=True, exist_ok=True)
        for tag in periods:
            s, e = dates(tag)
            name = f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}_{tag_}.geojson"
            if (target / name).is_file():
                continue
            src = BASE / "raw_detections" / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
            if not src.is_file():
                print(f"    {tag}: no raw detections at {src}")
                continue
            rc |= run([sys.executable, str(CORE / "postprocess.py"), str(src),
                       "--t-main", f"{t_main:g}", "--t-iso", f"{t_iso:g}",
                       "--k", str(NEIGHBOURS), "--D", f"{ISOLATION_KM:g}"], dry)
            # >> postprocess.py writes next to its input; the driver moves the file into
            # >> its folder.
            produced = src.with_name(name)
            if produced.is_file() and not dry:
                produced.rename(target / name)
    return rc


# >> STAGE persist-detections (executed, WHOLE_HISTORY): runs core/persistence.py over all
# >> years and quarters: a site is confirmed if detected in k=2 of a 2-year window; writes
# >> the cumulative detections (with onset/status) and patch_diffs/ used by
# >> mask-quarterly. Not skipped when outputs exist: it always recomputes from the full
# >> history.
def stage_persist_detections(periods, dry) -> int:
    years = [t for t in periods if Period.parse(t).is_annual]
    quarters = [t for t in periods if not Period.parse(t).is_annual]
    # --base explicitly, though persistence.py has a default: that default is
    # "../data/outputs/<model>", relative to the caller's cwd, so it only
    # resolves when run from core/. BASE is absolute and works from anywhere.
    return run([sys.executable, str(CORE / "persistence.py"),
                "--base", str(BASE),
                "--years", *years, "--quarters", *quarters, "--dissolve"], dry)


# >> Folder names that sam2_mask.py will have created for one period (derived, not
# >> searched): annual -> basin LOOSE run + Andes run; quarter -> the growth_<tag> run.
# >> Used by stage_cog.
def run_dirs(tag: str) -> List[Path]:
    """SAM2 run directories for one period.

    An annual period is segmented twice, basin and andes supplemental; a quarter
    once, from patch_diffs. sam2_mask.py names each directory after the detections
    file that prompted it, so the names are a function of the period.
    """
    period = Period.parse(tag)
    if period.is_annual:
        span = period.date_span
        return [SAM2 / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{span}_{LOOSE_TAG}",
                SAM2 / f"andes_supplemental_{MODEL}_{ANDES_TAG}_{span}"]
    return [SAM2 / f"Amazon_ACA_{MODEL}_growth_{tag}"]


# >> STAGE cog (executed): runs core/sam2_build_cog.py on each SAM2 run folder to mosaic
# >> the per-tile masks into per-(UTM zone x 8-deg lat band) COGs in <run>/cog_outputs/.
# >> Idempotent: skips a folder whose cog_outputs already holds any *utm*.tif. Gotcha:
# >> "any" file counts, so a run that crashed half-way is treated as done. Only masks are
# >> built (no --raster_types), so no logit mosaics come from here.
def stage_cog(periods, dry) -> int:
    """COG the run directories for the given periods.

    Named rather than discovered: data/outputs/sam2 also holds prior vintages and
    test output, and a stage that walks the directory has to recognise those to
    leave them alone.
    """
    rc = 0
    for tag in periods:
        for d in run_dirs(tag):
            if not d.is_dir():
                print(f"    {tag}: no run directory {d.name}")
                continue
            # >> Idempotence check (see gotcha above).
            if glob.glob(str(d / "cog_outputs" / "*utm*.tif")):
                continue
            if not glob.glob(str(d / "*-msk.tif")):
                print(f"    {tag}: {d.name} holds no mask tiles")
                continue
            rc |= run([sys.executable, str(CORE / "sam2_build_cog.py"), str(d)], dry)
    return rc


# >> Regex capturing the band group from a COG name, e.g. "utm21_lat_-16_-8".
#: Band group in a per-band mask COG name, e.g. utm21_lat_-16_-8.
GROUP_RE = re.compile(
    r"^mining_mask_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(utm\S+?)_epsg4326\.tif$")


# >> STAGE persist-masks (executed, WHOLE_HISTORY): finds every band group present in any
# >> SAM2 run's cog_outputs, runs core/sam2_persistence.py per group (onset raster:
# >> bounded_growth attribution to confirmed detections), then --mosaic to merge all
# >> groups into persistence_masks/amazon_basin_mining_scar_masks.tif. Always recomputes.
# >> Output check: every COG name must parse to a group, else hard stop. Gotcha: --years
# >> is not passed, so sam2_persistence uses its default 2018-2025; a new annual period
# >> (e.g. 2026) is silently left out until that default changes.
def stage_persist_masks(periods, dry) -> int:
    quarters = [t for t in periods if not Period.parse(t).is_annual]
    # mining_mask_<start>_<end>_<group>_epsg4326.tif. Matched by pattern rather
    # than by splitting on "_": the group name contains underscores itself, and a
    # wrong split yields a plausible string that matches nothing downstream.
    # >> Gotcha: globs ALL folders under data/outputs/sam2, including old vintages and
    # >> tests (stage_cog avoids exactly that by naming folders).
    names = [p.name for p in SAM2.glob("*/cog_outputs/mining_mask_*utm*.tif")]
    groups = sorted({m.group(1) for m in (GROUP_RE.match(n) for n in names) if m})
    unparsed = [n for n in names if not GROUP_RE.match(n)]
    if unparsed or not groups:
        raise SystemExit(
            f"could not parse band group from {len(unparsed)} filename(s), "
            f"e.g. {unparsed[:2]}; got {len(groups)} groups")
    rc = 0
    # >> One sam2_persistence run per band group (groups are independent).
    for g in groups:
        rc |= run([sys.executable, str(CORE / "sam2_persistence.py"),
                   "--group", g, "--quarters", *quarters,
                   "--outdir", str(SAM2 / "persistence_masks")], dry)
    rc |= run([sys.executable, str(CORE / "sam2_persistence.py"), "--mosaic",
               str(SAM2 / "persistence_masks"
                   / "amazon_basin_mining_scar_masks.tif"),
               "--outdir", str(SAM2 / "persistence_masks")], dry)
    return rc


# >> STAGE stage (executed, WHOLE_HISTORY): publishing for the AMW website/data release;
# >> not part of our research.
def stage_stage(periods, dry) -> int:
    return run([sys.executable, str(SCRIPTS / "stage_outputs.py"),
                "--periods", *periods], dry)


# >> STAGE manifest (executed, WHOLE_HISTORY; only prints a reminder): publishing for the
# >> AMW website/data release; not part of our research.
def stage_manifest(periods, dry) -> int:
    print("    MANIFEST.yaml is hand-maintained; update `updated`, `periods`,")
    print("    and any path_map changes. See data/outputs/MANIFEST.yaml.")
    return 0


# >> Stage name -> function that EXECUTES it (non-HUMAN stages).
RUNNERS = {"concat": stage_concat, "filter": stage_filter,
           "postprocess": stage_postprocess,
           "persist-detections": stage_persist_detections,
           "cog": stage_cog, "persist-masks": stage_persist_masks,
           "stage": stage_stage, "manifest": stage_manifest}


# >> CLI entry: parse stage names and periods, validate them all first, then run each
# >> stage in the order given (not re-sorted). Stops at the first stage that returns
# >> non-zero.
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"one or more of: {', '.join(ORDER)}")
    ap.add_argument("--periods", nargs="+", default=None,
                    help=("Periods this run works on, e.g. --periods Q326, or "
                          "--periods 2026 Q127. Each must be a member of "
                          "ALL_CURRENT_PERIODS in core/pipeline_config.py. No default: "
                          "a stale list is the one silent failure here."))
    ap.add_argument("--all", action="store_true", dest="use_all",
                    help="Work on every period in ALL_CURRENT_PERIODS (full rebuild)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the stage order")
    args = ap.parse_args()

    # >> --list (or no stage names): print the stage order and who runs each.
    if args.list or not args.stages:
        print("stages, in dependency order:\n")
        for s in ORDER:
            who = "HUMAN (prints commands)" if s in HUMAN else "pipeline"
            print(f"  {s:<20} {who}")
        print("\nmask-quarterly depends on persist-detections: its prompts are "
              "patch_diffs/.")
        return

    # Validate every name first: a bad stage late in the list would otherwise
    # abort after earlier stages had already run for real.
    # >> Refuse unknown names before running anything.
    unknown = [s for s in args.stages if s not in ORDER]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; see --list")

    # review-config and the whole-history stages read ALL_CURRENT_PERIODS
    # directly, so only the remaining stages need --periods.
    # >> Stages that actually consume --periods (i.e. not whole-history, not
    # >> review-config).
    consumers = [s for s in args.stages
                 if s not in WHOLE_HISTORY and s != "review-config"]

    if args.use_all:
        working = list(ALL_CURRENT_PERIODS)
    elif not consumers:
        working = list(ALL_CURRENT_PERIODS)      # not consumed by these stages
    elif args.periods:
        # Membership is required, not advisory: the whole-history stages read
        # ALL_CURRENT_PERIODS, so a period outside it silently never reaches the
        # product. Better to refuse than to half-run it.
        # >> Every requested period must be in ALL_CURRENT_PERIODS
        # >> (core/pipeline_config.py).
        stray = [p for p in args.periods if p not in ALL_CURRENT_PERIODS]
        if stray:
            raise SystemExit(
                f"period(s) {stray} are not in ALL_CURRENT_PERIODS.\n"
                f"  Add them to core/pipeline_config.py first -- see "
                f"`pipeline.py review-config`.\n"
                f"  Without that, persist-detections / persist-masks / stage "
                f"cannot see them.")
        working = list(args.periods)
    else:
        raise SystemExit(
            f"--periods is required for {consumers} (or --all for every period).\n"
            f"  It must name periods listed as ALL_CURRENT_PERIODS in "
            f"core/pipeline_config.py,\n"
            f"  which currently holds {len(ALL_CURRENT_PERIODS)}: "
            f"{' '.join(ALL_CURRENT_PERIODS)}\n"
            f"  e.g. --periods {ALL_CURRENT_PERIODS[-1]:<10} one period\n"
            f"       {'--all':<20} a full rebuild\n"
            f"  `pipeline.py review-config` prints this list and what depends "
            f"on it.")

    # >> Main loop: stages run in the order typed on the command line.
    for s in args.stages:
        print(f"\n=== {s}", flush=True)
        # >> WHOLE_HISTORY stages get the full list; others get the working periods.
        periods = ALL_CURRENT_PERIODS if s in WHOLE_HISTORY else working
        # >> HUMAN stage: print its commands and move on (nothing is executed).
        if s in HUMAN:
            # These are long VM jobs, or outward-facing. Printing is the only
            # thing this driver does with them, so it needs no flag to ask for it.
            for line in EMITTERS[s](periods):
                print(line, flush=True)
            continue
        # >> Executed stage: run it; a non-zero code aborts the remaining stages.
        rc = RUNNERS[s](periods, args.dry_run)
        if rc:
            raise SystemExit(f"{s} failed (rc={rc}); not continuing")


if __name__ == "__main__":
    main()
