#!/usr/bin/env bash
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
#
# run_staged.sh — run periods ONE AT A TIME with a cooldown in between.
#
# WHY: the audit of 2026-08-04 (docs/RUN2_SETUP_AND_AUDIT.md §3) found no CPU cap, no
# memory cap and no thermal safeguard anywhere in the codebase, and run_overnight.sh's
# `caffeinate -im` actively prevents the machine from ever dropping to a low-power state.
# This script duty-cycles the work instead: one period (~12 min monthly, ~28 min quarterly
# at 80 tiles), then an idle cooldown, then the next. Peak junction temperature — not
# total energy — is what damages hardware.
#
# WHY ONE PERIOD PER INVOCATION: panel.py accumulates a period's rows in memory and writes
# the parquet only at the end (panel.py:155, :236). A period boundary is therefore the only
# point at which stopping costs nothing. Cooling between periods is free; interrupting
# mid-period throws away up to 28 minutes of inference.
#
# USAGE — nothing needs to be exported; every default is correct.
#
#   bash timeseries/run_staged.sh 2025-01 2025-02 2025-03 q2025Q1   # run a batch
#   bash timeseries/run_staged.sh --dry-run 2025-01                 # print config, do nothing
#   bash timeseries/run_staged.sh --force 2025-01 q2025Q1           # rebuild existing panels
#
# Period labels are exactly what run_trials.py --months accepts (verified at
# run_trials.py:110-113): monthly as YYYY-MM, quarterly as qYYYYQN.
#
# ENVIRONMENT (all optional — override only if you want something non-default)
#   AMW_VENV=~/venvs/amw          venv location (outside the repo, survives repo moves)
#   AMW_OUT_ROOT=<repo>/data/timeseries_run2      panels, tile stats, detections, logs
#   AMW_CACHE_ROOT=<repo>/data/timeseries/cache   tile cache (gitignored; run 1's lives here)
#   AMW_YEAR=2025
#   AMW_REGION=<repo>/data/boundaries/la_pampa_corridor.geojson   the AOI to score
#   AMW_MODEL=<repo>/models/48px_v4.10b-...-ensemble.h5           the classifier
#   AMW_THREADS=4                 CPU thread budget (read by resource_limits.py)
#   AMW_WORKERS=4                 --max_workers (concurrent EE requests AND batch size)
#   AMW_COOLDOWN=600              MINIMUM idle seconds between periods
#   AMW_COOLDOWN_PCT=75           and at least this % of the period's own runtime
#                                 (duty cycle: 28 min of work -> 21 min of rest)
#   AMW_NICE=10                   nice level
#
# RESUME: safe to re-run at any time. run_trials.py:123-126 skips periods whose parquet
# already exists; cached tiles are reused. Ctrl-C between periods loses nothing.

# >> FILE OVERVIEW: fork-original (not upstream). Overnight launcher for the timeseries
# >> panels.
# >> Runs run_trials.py once per period label, one at a time, with a rest in between (duty
# >> cycle) so the laptop cools down. Checks the venv / Python version, finds
# >> EE_PROJECT, lowers process priority, logs to <out_root>/staged_<timestamp>.log.
# >> Header lines 3-40 above double as the usage text (printed by sed when no args given).
# >> set -e: stop on any error; -u: using an unset variable is an error; pipefail: a pipe
# >> fails if any stage fails.
set -euo pipefail

# >> REPO = parent of this script's folder, as an absolute path.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# >> Each setting: the environment variable if set, else the default after ':-'.
VENV="${AMW_VENV:-$HOME/venvs/amw}"
OUT_ROOT="${AMW_OUT_ROOT:-$REPO/data/timeseries_run2}"
# Everything lives inside the repo now. The cache was only ever outside because the repo
# sat in iCloud (RUN2_SETUP_AND_AUDIT.md §1.4); since the 2026-08-07 relocation to
# ~/RA_Raji there is no reason to split it. data/timeseries/cache/ is already gitignored,
# and pointing at it reuses run 1's 2,560 cached tiles instead of re-fetching them.
CACHE_ROOT="${AMW_CACHE_ROOT:-$REPO/data/timeseries/cache}"
YEAR="${AMW_YEAR:-2025}"
# The AOI. run_trials.py:78 defaults to la_pampa_corridor.geojson; setting it explicitly
# here means the choice is visible in the run banner and the log, not buried in a default.
REGION="${AMW_REGION:-$REPO/data/boundaries/la_pampa_corridor.geojson}"
MODEL="${AMW_MODEL:-$REPO/models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5}"
THREADS="${AMW_THREADS:-4}"
WORKERS="${AMW_WORKERS:-4}"
LIMIT_TILES="${AMW_LIMIT_TILES:-}"       # smoke test: only the first N tiles of the AOI
# >> Rest rule: max(600 s floor, 75% of the period's runtime); see cooldown_for below.
COOLDOWN="${AMW_COOLDOWN:-600}"          # minimum idle seconds between periods
COOLDOWN_PCT="${AMW_COOLDOWN_PCT:-75}"   # AND at least this % of the period's own runtime
NICE_LEVEL="${AMW_NICE:-10}"

DRY_RUN=0
FORCE=0
ANALYSE=0
# >> Parse leading flags; whatever remains are period labels (e.g. 2025-01, q2025Q1).
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1;   shift ;;   # rebuild panels even if the parquet exists
    --analyse|--analyze) ANALYSE=1; shift ;;  # run analysis.py after the batch
    --)        shift; break ;;
    -*)        echo "unknown flag: $1" >&2; exit 2 ;;
    *)         break ;;
  esac
done
# >> Optional flags are stored as arrays so they expand to zero or more separate words.
# NOTE: expanded below as ${arr[@]+"${arr[@]}"}, not "${arr[@]}". macOS ships bash 3.2,
# where expanding an EMPTY array under `set -u` raises "unbound variable". The `+`
# form yields nothing when the array is unset/empty and is safe in bash 3.2 and 5.x.
FORCE_ARG=()
[ "$FORCE" = "1" ] && FORCE_ARG=(--force)
LIMIT_ARG=()
[ -n "$LIMIT_TILES" ] && LIMIT_ARG=(--limit_tiles "$LIMIT_TILES")

# >> No period labels -> print the usage header (lines 3-40 of this file) and exit 2.
if [ $# -eq 0 ]; then
  sed -n '3,40p' "$0"
  exit 2
fi
PERIODS=("$@")

# --------------------------------------------------------------- environment
# >> Stop if the venv is missing (default ~/venvs/amw).
if [ ! -f "$VENV/bin/activate" ]; then
  echo "FATAL: no venv at $VENV. Build it per docs/CODE_WALKTHROUGH.md §0.3." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# >> Python gate: must be exactly 3.11 (TensorFlow 2.14 wheels stop at cp311).
# Verification gate — '(venv)' in a prompt proves nothing (RUN2_SETUP_AND_AUDIT.md §1.5).
PY_PATH="$(command -v python)"
PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PY_VER" != "3.11" ]; then
  echo "FATAL: python is $PY_VER at $PY_PATH; 3.11 required (tensorflow==2.14.0 has no" >&2
  echo "       wheels above cp311). Activate $VENV and retry." >&2
  exit 1
fi

# >> EE_PROJECT = Google Cloud project billed for Earth Engine. A value from the shell
# >> wins; otherwise the EE_PROJECT line of .env is parsed (sed strips key, '=', spaces;
# >> tr strips quotes, tabs and Windows CR).
# EE_PROJECT: .env is NOT read by Python (gee.py:20 falls back to upstream's 'earthindex').
# Parse it out of .env without sourcing — the file is not valid shell (spaces around '=').
if [ -z "${EE_PROJECT:-}" ] && [ -f "$REPO/.env" ]; then
  EE_PROJECT=$(sed -n 's/^EE_PROJECT[[:space:]]*=[[:space:]]*//p' "$REPO/.env" \
               | head -1 | tr -d '"'"'"' \t\r')
  export EE_PROJECT
fi
# >> ':?' = abort with this message if EE_PROJECT is still empty.
: "${EE_PROJECT:?EE_PROJECT not set — put it in .env or export it}"

# >> The region GeoJSON and the model .h5 must exist.
for F in "$REGION" "$MODEL"; do
  [ -f "$F" ] || { echo "FATAL: not found: $F" >&2; exit 1; }
done

# >> Export the thread budget so run_trials.py -> resource_limits.py can read it.
export AMW_THREADS="$THREADS"

# Directory creation can fail with EPERM inside iCloud-managed folders (the same file
# provider refusal that blocks venv creation — RUN2_SETUP_AND_AUDIT.md §1.4). Fail with an
# actionable message rather than a bare mkdir error.
for D in "$CACHE_ROOT/$YEAR" "$OUT_ROOT"; do
  if ! mkdir -p "$D" 2>/dev/null; then
    echo "FATAL: cannot create $D" >&2
    echo "" >&2
    echo "If this path is under ~/Documents it is iCloud-managed and the file provider" >&2
    echo "may refuse new directories. Put the run outputs outside iCloud:" >&2
    echo "" >&2
    echo "  mkdir -p ~/amw_runs" >&2
    echo "  AMW_OUT_ROOT=~/amw_runs/timeseries_run2 bash timeseries/run_staged.sh $*" >&2
    echo "" >&2
    exit 1
  fi
done
# >> LOG FILE: one per invocation, staged_<YYYYmmdd_HHMMSS>.log in OUT_ROOT (appended via
# >> tee).
LOG="$OUT_ROOT/staged_$(date +%Y%m%d_%H%M%S).log"

# Process priority: taskpolicy -b sets macOS background QoS (on Apple Silicon this steers
# work toward efficiency cores); nice is the portable fallback. Both are best-effort.
# >> macOS: taskpolicy -b (background QoS); elsewhere nice -n 10 (lower CPU priority).
if command -v taskpolicy >/dev/null 2>&1; then
  PRIO=(taskpolicy -b)
else
  PRIO=(nice -n "$NICE_LEVEL")
fi

# ---------------------------------------------------------------- thermal probe
# `pmset -g therm` reports CPU_Speed_Limit without sudo on macOS. If the command is absent
# or its output does not parse, we degrade to "unknown" and never block on it.
# >> thermal_pct: prints CPU_Speed_Limit as a number (100 = full speed, < 100 =
# >> throttled), or nothing if pmset (macOS power tool) is missing or has no such line.
thermal_pct() {
  local out
  out="$(pmset -g therm 2>/dev/null || true)"
  echo "$out" | sed -n 's/.*CPU_Speed_Limit[^0-9]*\([0-9][0-9]*\).*/\1/p' | head -1
}

# >> report_thermal: one readable thermal line for the log (unknown / THROTTLED /
# >> nominal).
report_thermal() {
  local pct; pct="$(thermal_pct)"
  if [ -z "$pct" ]; then
    echo "thermal: unknown (pmset -g therm unavailable or unparsed)"
  elif [ "$pct" -lt 100 ]; then
    echo "thermal: THROTTLED — CPU_Speed_Limit=${pct}%"
  else
    echo "thermal: nominal (CPU_Speed_Limit=100%)"
  fi
}

# Cooldown length.
#
# `pmset -g therm` exposes CPU_Speed_Limit on Intel Macs but generally reports nothing
# parseable on Apple Silicon, and there is no no-sudo temperature API there. So the PRIMARY
# mechanism is duration-proportional and needs no sensor at all: rest for
# AMW_COOLDOWN_RATIO x the time the period just took, floored at AMW_COOLDOWN. A 12-min
# month and a 28-min quarter therefore get proportionate rest automatically.
#
# The thermal probe, when it does work, only ever EXTENDS this — it can never shorten it.
# >> cooldown_for SECONDS_WORKED: sleep max(AMW_COOLDOWN, worked x AMW_COOLDOWN_PCT /
# >> 100), e.g. 28 min of work -> 21 min rest; 12 min -> 10 min floor. Then, while pmset
# >> reports throttling, sleep 120 s more, up to 3x the planned rest in total. Integer
# >> maths only.
# >> Gotcha: the comment above says AMW_COOLDOWN_RATIO; the real variable is
# >> AMW_COOLDOWN_PCT.
cooldown_for() {
  local worked="${1:-0}" want waited=0 max pct
  # bash has no float arithmetic; ratio is applied as percent
  want=$(( worked * COOLDOWN_PCT / 100 ))
  [ "$want" -lt "$COOLDOWN" ] && want="$COOLDOWN"
  # >> Hard cap on total rest = 3 x the planned rest.
  max=$(( want * 3 ))

  echo "    cooling down ${want}s (worked ${worked}s, ratio ${COOLDOWN_PCT}%, floor ${COOLDOWN}s) …" \
    | tee -a "$LOG"
  sleep "$want"; waited="$want"

  # >> Thermal extension loop: it can only lengthen the rest, never shorten it.
  while [ "$waited" -lt "$max" ]; do
    pct="$(thermal_pct)"
    [ -z "$pct" ] && return 0            # cannot measure -> proportional cooldown only
    [ "$pct" -ge 100 ] && return 0
    echo "    still throttled at ${pct}% after ${waited}s — extending" | tee -a "$LOG"
    sleep 120; waited=$((waited + 120))
  done
  echo "    WARNING: still throttled after ${waited}s; continuing anyway" | tee -a "$LOG"
}

# --------------------------------------------------------------------- banner
# >> Banner: config summary to the screen and the log (tee -a appends to the log file).
{
  echo "=== staged run started $(date) ==="
  echo "repo=$REPO"
  echo "python=$PY_PATH ($PY_VER)"
  echo "EE_PROJECT=$EE_PROJECT"
  echo "out_root=$OUT_ROOT"
  echo "cache=$CACHE_ROOT/$YEAR"
  echo "region=$(basename "$REGION")"
  echo "model=$(basename "$MODEL")"
  echo "year=$YEAR  threads=$THREADS  max_workers=$WORKERS${LIMIT_TILES:+  limit_tiles=$LIMIT_TILES}"
  echo "cooldown=${COOLDOWN}s  priority=${PRIO[*]}"
  echo "periods=${PERIODS[*]}  (${#PERIODS[@]} total)"
  report_thermal
} | tee -a "$LOG"

# >> --dry-run stops after the banner.
if [ "$DRY_RUN" = "1" ]; then
  echo "--dry-run: would execute the above, one period at a time. Exiting." | tee -a "$LOG"
  exit 0
fi

# ----------------------------------------------------------------------- run
# >> FAILED collects the labels whose run_trials.py exited non-zero.
FAILED=()
# >> Loop over period labels in the order given on the command line.
for i in "${!PERIODS[@]}"; do
  P="${PERIODS[$i]}"
  N=$((i + 1))
  echo "" | tee -a "$LOG"
  echo "--- [$N/${#PERIODS[@]}] period $P  $(date '+%H:%M:%S') ---" | tee -a "$LOG"
  report_thermal | tee -a "$LOG"

  # >> Wall-clock start (seconds since 1970) to time the period.
  START=$(date +%s)
  # >> Run ONE period: priority prefix + run_trials.py --months <label>; all output to
  # >> LOG. run_trials.py itself skips the period if its parquet already exists
  # >> (resume-safe).
  if "${PRIO[@]}" python "$REPO/timeseries/run_trials.py" \
        --region "$REGION" \
        --model "$MODEL" \
        --out_root "$OUT_ROOT" \
        --cache_dir "$CACHE_ROOT/$YEAR" \
        --year "$YEAR" \
        --months "$P" \
        --max_workers "$WORKERS" \
        ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} >>"$LOG" 2>&1; then
    ELAPSED=$(( $(date +%s) - START ))
    echo "    OK in $(( ELAPSED / 60 )) min" | tee -a "$LOG"
  else
    ELAPSED=$(( $(date +%s) - START ))
    echo "    FAILED after $(( ELAPSED / 60 )) min (see $LOG)" | tee -a "$LOG"
    FAILED+=("$P")
  fi
  report_thermal | tee -a "$LOG"

  # >> Rest between periods, but not after the last one.
  if [ "$N" -lt "${#PERIODS[@]}" ]; then
    cooldown_for "$ELAPSED"
  fi
done

echo "" | tee -a "$LOG"
# >> Any failure -> exit 1; re-running the same command retries only unfinished periods.
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== done WITH FAILURES: ${FAILED[*]} ===" | tee -a "$LOG"
  echo "Re-run the same command — finished periods are skipped, cached tiles reused." | tee -a "$LOG"
  exit 1
fi
echo "=== all ${#PERIODS[@]} periods done $(date) ===" | tee -a "$LOG"

# --- optional: the statistical comparison over every panel in OUT_ROOT --------
# Not run by default because it pools ALL panels present (analysis.py:94 globs with no
# filter). Only meaningful once the periods you want compared are all built.
# >> --analyse: runs analysis.py over <out_root>/panels and tiles; writes
# >> analysis/REPORT.md.
if [ "$ANALYSE" = "1" ]; then
  echo "--- analysis $(date '+%H:%M:%S') ---" | tee -a "$LOG"
  if python "$REPO/timeseries/analysis.py" \
        --panels_dir "$OUT_ROOT/panels" \
        --tiles_dir  "$OUT_ROOT/tiles" \
        --out_dir    "$OUT_ROOT/analysis" >>"$LOG" 2>&1; then
    echo "    report: $OUT_ROOT/analysis/REPORT.md" | tee -a "$LOG"
  else
    echo "    ANALYSIS FAILED (see $LOG)" | tee -a "$LOG"
    exit 1
  fi
fi

echo "Log: $LOG"
