#!/usr/bin/env bash
# PROVENANCE: FORK-ORIGINAL — fork-only file (RA timeseries project); not in upstream earthrise-media/mining-detector
#
# Standard upstream sync with auto-resolution of "we archived it" conflicts.
#
# Policy encoded here: any merge conflict of type DU (Deleted by Us, modified
# Upstream) is resolved as "stay deleted" — those are exactly the files this
# fork moved to _archive/. Everything else (both sides edited, or upstream
# deleted something we changed) is a REAL conflict and is left for human eyes.
#
# Usage:  bash scripts/sync_upstream.sh      (from anywhere inside the repo)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: uncommitted changes present. Commit or stash first." >&2
    exit 1
fi

git fetch upstream
git branch "backup-before-upstream-$(date +%Y-%m-%d)" 2>/dev/null \
    && echo "Safety branch: backup-before-upstream-$(date +%Y-%m-%d)" || true

if git merge --no-edit upstream/main; then
    echo "Merged clean — no conflicts."
else
    echo
    echo "Auto-resolving archived-file conflicts (deleted by us / modified upstream):"
    git status --porcelain | while IFS= read -r line; do
        st="${line:0:2}"
        p="${line:3}"
        if [ "$st" = "DU" ]; then
            git rm -q -- "$p"
            echo "  kept deleted: $p"
        fi
    done
    left="$(git diff --name-only --diff-filter=U)"
    if [ -n "$left" ]; then
        echo
        echo "REAL conflicts — resolve by hand, then 'git add <file>' and 'git commit':"
        echo "$left"
        exit 2
    fi
    git commit --no-edit
    echo "All conflicts auto-resolved; merge committed."
fi

echo
echo "Post-sync audit:  python3 scripts/provenance_report.py"
echo "  -> check the 'archive candidates' section for new upstream training junk."
