# Git playbook for this fork

How to keep this fork current with `earthrise-media/mining-detector` **without**
resurrecting the files we archived, and what is / isn't visible to the upstream
owners. Written 2026-07-27, after upstream merged PR #29 (`2026dense_embed` → `main`).

---

## 1. The mental model

You have **one local repo** with **two remotes** (a remote = a named URL git can
talk to):

| Remote | URL | Role |
|---|---|---|
| `origin` | `github.com/RubyAnn777/AMW-mining-detector` | **your fork** — you push here |
| `upstream` | `github.com/earthrise-media/mining-detector` | **their repo** — you only *fetch* from here |

Three facts that answer most confusions:

1. **A fork never auto-updates.** GitHub copied their repo once, at fork time.
   New upstream commits arrive in your repo only when *you* run `git fetch upstream`.
2. **`git pull` on `main` only talks to `origin`.** That's why it said
   "Already up to date" — your fork's `main` on GitHub hadn't changed. It never
   looks at `upstream` unless you tell it to.
3. **Branches are per-remote.** `upstream/main` and your `main` are different
   branches that merely share history. Fetching upstream updates your local
   *copy* of their branches (`remotes/upstream/...`); it touches none of your files.

## 2. Visibility: what can the upstream owners see?

- **Commits you don't push** exist only on your machine. Invisible to everyone.
- **Commits pushed to `origin`** are on *your* fork. If your fork is public,
  anyone (including upstream owners) *could* browse it, but nothing appears in
  their repo, their history, or their notifications. Forks are expected; nobody
  audits them.
- **The only way your work enters their repo** is if you deliberately open a
  Pull Request. Nothing we do here does that.
- **Pushing to upstream is physically impossible** in this repo: the push URL is
  set to `DISABLE` (`git remote -v` shows `upstream DISABLE (push)`). This is a
  remote-level setting, so it covers **every** branch of upstream — `main`,
  `2026dense_embed`, all of them. No per-branch rule needed after PR #29.
- License note: upstream is MIT-licensed, so keeping and modifying a fork
  (even privately) is explicitly permitted.

## 3. Why we do NOT `git merge upstream/main`

A merge replays *all* upstream history onto your branch — including files you
deleted/archived. Git treats your deletion and their (non-)changes as a
conflict or, worse, silently re-adds files they touched. **Every archived file
they later edit would reappear.** That is exactly the blind spot you worried
about, and it's structural, not bad luck.

Instead we use **path checkout**: copy *only the files we still track* from
their branch into ours. Files you never name can never come back.

## 4. The routine (run whenever upstream might have moved)

```bash
# 0. Safety net (optional but cheap): snapshot your current state
git branch backup-$(date +%F)

# 1. Download their new commits (updates remotes/upstream/*, touches no files)
git fetch upstream

# 2. See what changed in the folders we track — read before you take
git diff --stat HEAD upstream/main -- core/ models/ scripts/ utils/
git log --oneline -10 upstream/main

# 3. Take ONLY those paths (add/remove paths as our scope changes)
git checkout upstream/main -- core/ models/ scripts/ utils/

# 4. Review, then commit on YOUR fork
git status
git diff --staged --stat
git commit -m "Sync core/ models/ scripts/ utils/ from upstream/main (<hash>)"
git push origin main        # optional; only if you want it on your GitHub fork
```

Notes:

- `git checkout upstream/main -- <path>` **overwrites your local edits inside
  that path**. We deliberately keep our own code *outside* upstream folders
  (`timeseries/`, `time_tiles.py`, this file, the card) so syncing never
  clobbers it. If you ever must edit an upstream file, commit first, then diff
  after checkout.
- Since PR #29, **track `upstream/main`**, not `upstream/2026dense_embed`. The
  two now differ only in README wording; future work lands on `main`.
- Their root `README.md` is *not* in our sync list — ours is customized.
  If you want their README changes: `git diff upstream/main -- README.md` and
  cherry-pick by hand.

## 5. State as of 2026-07-27 (audit result)

- `core/` — **byte-identical to `upstream/main`** after syncing `core/README.md`
  (their Jul-24 doc edits were the only drift; your Jul-24 copy caught
  everything else, including `model_evaluation.ipynb` and the v4.10b weights + t0.4 config).
- `models/` — your v4.10b ensemble + config match upstream; the many deleted
  single-model `.h5` files are your intentional archiving. Correct as is.
- `scripts/`, `utils/` — no upstream changes since your copies (checked Jul 18 → now).
- Root `README.md`, `.gitignore` — yours, customized; keep.

## 6. One-line cheat sheet

| Want | Command |
|---|---|
| See their new commits | `git fetch upstream && git log --oneline main..upstream/main` |
| What would change in core/ | `git diff --stat HEAD upstream/main -- core/` |
| Take their core/ | `git checkout upstream/main -- core/` |
| Undo an un-committed take | `git restore --staged --worktree core/` then `git checkout HEAD -- core/` |
| Where am I | `git status` / `git branch -a` |
| Panic button | `git checkout backup-<date>` |
