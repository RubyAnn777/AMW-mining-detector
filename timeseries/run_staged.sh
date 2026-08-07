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
# USAGE
#   bash timeseries/run_staged.sh 2025-01 2025-02 2025-03 q2025Q1
#   bash timeseries/run_staged.sh --dry-run 2025-01 q2025Q1
#
# Period labels are exactly what run_trials.py --months accepts (verified at
# run_trials.py:110-113): monthly as YYYY-MM, quarterly as qYYYYQN.
#
# ENVIRONMENT (all optional, defaults shown)
#   AMW_VENV=~/venvs/amw          venv location (OUTSIDE the repo — iCloud, see §1.4)
#   AMW_OUT_ROOT=<repo>/data/timeseries_run2
#   AMW_CACHE_ROOT=~/amw_cache    tile cache — MUST be outside iCloud
#   AMW_YEAR=2025
#   AMW_THREADS=4                 CPU thread budget (read by resource_limits.py)
#   AMW_WORKERS=4                 --max_workers (concurrent EE requests AND batch size)
#   AMW_COOLDOWN=600              MINIMUM idle seconds between periods
#   AMW_COOLDOWN_PCT=75           and at least this % of the period's own runtime
#                                 (duty cycle: 28 min of work -> 21 min of rest)
#   AMW_NICE=10                   nice level
#
# RESUME: safe to re-run at any time. run_trials.py:123-126 skips periods whose parquet
# already exists; cached tiles are reused. Ctrl-C between periods loses nothing.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${AMW_VENV:-$HOME/venvs/amw}"
OUT_ROOT="${AMW_OUT_ROOT:-$REPO/data/timeseries_run2}"
CACHE_ROOT="${AMW_CACHE_ROOT:-$HOME/amw_cache}"
YEAR="${AMW_YEAR:-2025}"
THREADS="${AMW_THREADS:-4}"
WORKERS="${AMW_WORKERS:-4}"
COOLDOWN="${AMW_COOLDOWN:-600}"          # minimum idle seconds between periods
COOLDOWN_PCT="${AMW_COOLDOWN_PCT:-75}"   # AND at least this % of the period's own runtime
NICE_LEVEL="${AMW_NICE:-10}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi

if [ $# -eq 0 ]; then
  sed -n '3,40p' "$0"
  exit 2
fi
PERIODS=("$@")

# --------------------------------------------------------------- environment
if [ ! -f "$VENV/bin/activate" ]; then
  echo "FATAL: no venv at $VENV. Build it per docs/CODE_WALKTHROUGH.md §0.3." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Verification gate — '(venv)' in a prompt proves nothing (RUN2_SETUP_AND_AUDIT.md §1.5).
PY_PATH="$(command -v python)"
PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PY_VER" != "3.11" ]; then
  echo "FATAL: python is $PY_VER at $PY_PATH; 3.11 required (tensorflow==2.14.0 has no" >&2
  echo "       wheels above cp311). Activate $VENV and retry." >&2
  exit 1
fi

# EE_PROJECT: .env is NOT read by Python (gee.py:20 falls back to upstream's 'earthindex').
# Parse it out of .env without sourcing — the file is not valid shell (spaces around '=').
if [ -z "${EE_PROJECT:-}" ] && [ -f "$REPO/.env" ]; then
  EE_PROJECT=$(sed -n 's/^EE_PROJECT[[:space:]]*=[[:space:]]*//p' "$REPO/.env" \
               | head -1 | tr -d '"'"'"' \t\r')
  export EE_PROJECT
fi
: "${EE_PROJECT:?EE_PROJECT not set — put it in .env or export it}"

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
LOG="$OUT_ROOT/staged_$(date +%Y%m%d_%H%M%S).log"

# Process priority: taskpolicy -b sets macOS background QoS (on Apple Silicon this steers
# work toward efficiency cores); nice is the portable fallback. Both are best-effort.
if command -v taskpolicy >/dev/null 2>&1; then
  PRIO=(taskpolicy -b)
else
  PRIO=(nice -n "$NICE_LEVEL")
fi

# ---------------------------------------------------------------- thermal probe
# `pmset -g therm` reports CPU_Speed_Limit without sudo on macOS. If the command is absent
# or its output does not parse, we degrade to "unknown" and never block on it.
thermal_pct() {
  local out
  out="$(pmset -g therm 2>/dev/null || true)"
  echo "$out" | sed -n 's/.*CPU_Speed_Limit[^0-9]*\([0-9][0-9]*\).*/\1/p' | head -1
}

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
cooldown_for() {
  local worked="${1:-0}" want waited=0 max pct
  # bash has no float arithmetic; ratio is applied as percent
  want=$(( worked * COOLDOWN_PCT / 100 ))
  [ "$want" -lt "$COOLDOWN" ] && want="$COOLDOWN"
  max=$(( want * 3 ))

  echo "    cooling down ${want}s (worked ${worked}s, ratio ${COOLDOWN_PCT}%, floor ${COOLDOWN}s) …" \
    | tee -a "$LOG"
  sleep "$want"; waited="$want"

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
{
  echo "=== staged run started $(date) ==="
  echo "repo=$REPO"
  echo "python=$PY_PATH ($PY_VER)"
  echo "EE_PROJECT=$EE_PROJECT"
  echo "out_root=$OUT_ROOT"
  echo "cache=$CACHE_ROOT/$YEAR"
  echo "year=$YEAR  threads=$THREADS  max_workers=$WORKERS"
  echo "cooldown=${COOLDOWN}s  priority=${PRIO[*]}"
  echo "periods=${PERIODS[*]}  (${#PERIODS[@]} total)"
  report_thermal
} | tee -a "$LOG"

if [ "$DRY_RUN" = "1" ]; then
  echo "--dry-run: would execute the above, one period at a time. Exiting." | tee -a "$LOG"
  exit 0
fi

# ----------------------------------------------------------------------- run
FAILED=()
for i in "${!PERIODS[@]}"; do
  P="${PERIODS[$i]}"
  N=$((i + 1))
  echo "" | tee -a "$LOG"
  echo "--- [$N/${#PERIODS[@]}] period $P  $(date '+%H:%M:%S') ---" | tee -a "$LOG"
  report_thermal | tee -a "$LOG"

  START=$(date +%s)
  if "${PRIO[@]}" python "$REPO/timeseries/run_trials.py" \
        --out_root "$OUT_ROOT" \
        --cache_dir "$CACHE_ROOT/$YEAR" \
        --year "$YEAR" \
        --months "$P" \
        --max_workers "$WORKERS" >>"$LOG" 2>&1; then
    ELAPSED=$(( $(date +%s) - START ))
    echo "    OK in $(( ELAPSED / 60 )) min" | tee -a "$LOG"
  else
    ELAPSED=$(( $(date +%s) - START ))
    echo "    FAILED after $(( ELAPSED / 60 )) min (see $LOG)" | tee -a "$LOG"
    FAILED+=("$P")
  fi
  report_thermal | tee -a "$LOG"

  if [ "$N" -lt "${#PERIODS[@]}" ]; then
    cooldown_for "$ELAPSED"
  fi
done

echo "" | tee -a "$LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== done WITH FAILURES: ${FAILED[*]} ===" | tee -a "$LOG"
  echo "Re-run the same command — finished periods are skipped, cached tiles reused." | tee -a "$LOG"
  exit 1
fi
echo "=== all ${#PERIODS[@]} periods done $(date) ===" | tee -a "$LOG"
echo "Log: $LOG"
