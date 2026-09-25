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

set -euo pipefail # set -e -> abprts execution on error, -u treats unset variables as error and exits, -o pipeline fails if any command fails
 cd "$(git rev-parse --show-toplevel)" # returns absolure ptath to the root dir of the git repo, and cd's to it

if [ -n "$(git status --porcelain)" ]; then # checks if working directory is clean (no uncommitted changes). If not, prints error and exits with code 1
    echo "ERROR: uncommitted changes present. Commit or stash first." >&2
    exit 1
fi

git fetch upstream # here we fetch the latest AMW changes without merging anything yet and create a backup branch
git branch "backup-before-upstream-$(date +%Y-%m-%d)" 2>/dev/null \ 
    && echo "Safety branch: backup-before-upstream-$(date +%Y-%m-%d)" || true

#attempt the merge 
if git merge --no-edit upstream/main; then
    echo "Merged clean — no conflicts."
else
    echo
    echo "Auto-resolving archived-file conflicts (deleted by us / modified upstream):"
    git status --porcelain | while IFS= read -r line; do # store each merge conflict line in variable line and process it
        st="${line:0:2}" # store status code for the file in variable st (first two characters of line)
        p="${line:3}" # store the file path in variable p (everything after the first three characters of line)
        if [ "$st" = "DU" ]; then # auto resolving DU conflicts - DU = deleted by us, modified upstream aka the ones we archived int he fork
            git rm -q -- "$p" # removes file and stages the removal for commit, q means quite and $p is the path 
            echo "  kept deleted: $p" # pastes the file name that was affected by the auto resolution
        fi
    done
    left="$(git diff --name-only --diff-filter=U)" # checking for remaining unresolved conflicts
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
