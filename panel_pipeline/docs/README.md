# Documents about the panel pipeline

Every fact has one living home. Start with `STATE.md`; the rest answers one question each.

The folder has five files at the top (the living ones) and four subfolders:
`code/` (how our scripts work), `theory/` (computer-science concepts), `cluster/` (Goethe)
and `history/` (old notes, not maintained).

| Question | Document |
|---|---|
| How do I run it, on a laptop or on Goethe? | `../README.md` |
| Where does the project stand, what did we decide, and how does Ruby work? | `STATE.md` |
| What is still open? | `open_design_decisions.Rmd` (principles P1–P6, decisions D1–D13, leftovers L1–L12) |
| What did we measure? | `findings.md` |
| The short version for colleagues | `team_brief.md` |
| How does the cluster run work, and what limits it? | `cluster/cluster_plan_goethe.md` (the R1–R5 block of 23 Oct comes first and wins) |
| How do I set up the Earth Engine service account? | `cluster/service_account_guide.md` |
| Why is assembly built the way it is? | `code/panel_assembly_strategy.md` (its "Update 21 Oct" overrides the body) |
| Do chips, tiles or subregions overlap? Chips or 240 m cells? | `open_design_decisions.Rmd` D5, D6 (scripts: `../checks/overlap/`) |
| What is a class / dataclass / property? | `theory/periods_explained.md` |
| What happens when I type `python plan.py ...`? | `theory/how_a_script_runs.md` |
| Threads, workers, the lock file, status.py | `code/workers_locks_and_status.md` |
| How do our scores compare with AMW's? | `../../amw_comparison/README.md`, `../../amw_comparison/referee_audit.md` |

## Rules for keeping this tidy

- A new decision goes into `STATE.md` §4 (with who and when); a new open question into the Rmd;
  a new measurement into `findings.md`. Don't start a new overview file.
- A note that stops being true moves to `history/` with a first line saying what replaced it.
- `code/` explains our scripts and must follow the code when it changes; `theory/` teaches
  concepts, whose line numbers may drift.

## history/

Kept for the reasoning, not maintained. Each file's first line says what replaced it.

| File | From | Still worth reading for |
|---|---|---|
| `handoff_manual.md` | 19 Oct | AMW's 13 stages (§5), variables still to build (§10) |
| `handoff_pipeline_redesign.md` | 15–19 Oct | nothing new (older copy of the handoff manual) |
| `stage_review.md` | 8 Oct | the first walk through AMW's stages |
| `round2_answers.md` | 8 Oct | overlap objections, SAM2 design, persistence ideas |
| `questions.md` | Ruby, 8 Oct | Ruby's own overlap and SAM2 notes |
| `run_design.md` | 8–15 Oct | routing principles (no typed paths, atomic writes) |
| `quarterly_and_parallel_runs.md` | 19 Oct | the 840-task layout, the review behind leftovers L1–L12 |
| `cluster_migration.md` | 19 Oct | packages, secrets, what the first cluster map got right and wrong |
| `detect_walkthrough.md` | 15 Oct | imagery and scoring blocks (A–C) explained line by line |
| `reading_the_notebook_test_region_2026.md` | 19 Oct | the per-month tables behind `findings.md` §1 |
