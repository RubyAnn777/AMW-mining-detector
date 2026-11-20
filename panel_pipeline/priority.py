"""Make a run's priority list, once: which tiles detect.py --priority 1 scores first.

    python panel_pipeline/priority.py --run <run> --rings 3            make it
    python panel_pipeline/priority.py --run <run> --rings 3 --dry-run  only show the counts
    python panel_pipeline/priority.py --run <run> --show               show the run's list

Priority 1 = every tile in which AMW has ever detected mining, plus `--rings` rings of
tiles around them; priority 2 = all other tiles. It reads what amw_comparison/history.py
wrote for this run (<run>/amw_comparison/history_tiles.parquet: AMW's raw detection files
mapped onto the run's tiles, with each tile's ring), so run that first:

    sbatch cluster/amw_history.sbatch <run>

Writes <run>/priority/tiles.csv and record.json (lib/priority.py). Made once: a run's list
never changes after it exists, because every task scored with --priority records which
list it used. To try other rings, use --dry-run; to really change it, the folder must be
moved aside by hand, and only before any task has used it (this script checks).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))   # our helpers

import priority as prio
from config import load_config
from provenance import environment, repo_path


def used_by(run_dir: Path) -> list[str]:
    """Task folders whose _status.json names a priority list (any list)."""
    out = []
    for status in (run_dir / "detect").glob("*/period=*/subregion=*/_status.json"):
        try:
            if "priority" in json.loads(status.read_text()):
                out.append(str(status.parent.relative_to(run_dir)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build(run_dir: Path, rings: int) -> tuple[pd.DataFrame, dict]:
    """The list (one row per tile of the run) and what goes into record.json about its source."""
    results = run_dir / "amw_comparison"
    path = results / "history_tiles.parquet"
    if not path.is_file():
        raise SystemExit(f"PRIORITY ERROR: {path} is missing; run "
                         f"`sbatch cluster/amw_history.sbatch {run_dir}` first")
    summary = json.loads((results / "history_summary.json").read_text())
    if "files" not in summary:
        raise SystemExit("PRIORITY ERROR: history_tiles.parquet was made from AMW's cumulative "
                         "record by an older history.py; run amw_history.sbatch again")
    if not 0 <= rings <= summary["max_ring"]:
        raise SystemExit(f"PRIORITY ERROR: --rings must be 0 to {summary['max_ring']} "
                         f"(history.py computed rings up to {summary['max_ring']})")
    history = pd.read_parquet(path, columns=["tile", "tile_key", "ring"])

    planned = pd.read_csv(run_dir / "plan" / "tiles.csv")
    keys = planned["tile_key"].drop_duplicates()
    numbers = keys.map(prio.number_of_key)
    missing = set(numbers) - set(history["tile"])
    extra = set(history["tile"]) - set(numbers)
    if missing or extra:
        raise SystemExit(f"PRIORITY ERROR: history_tiles.parquet does not match plan/tiles.csv "
                         f"({len(missing)} planned tiles missing, {len(extra)} unknown tiles); "
                         f"was it made for another run?")
    table = (pd.DataFrame({"tile_key": keys.to_numpy(), "tile": numbers.to_numpy()})
             .merge(history[["tile", "ring"]], on="tile", validate="one_to_one"))
    table["priority"] = (table["ring"] > rings).astype(int) + 1
    table = table.sort_values("tile", ignore_index=True)

    subs = planned.assign(tile=planned["tile_key"].map(prio.number_of_key)).merge(table[["tile", "priority"]], on="tile")
    per_sub = {str(int(n)): {str(p): int(c) for p, c in g["priority"].value_counts().sort_index().items()}
               for n, g in subs.groupby("subregion")}
    source = {
        "history_tiles": repo_path(path),
        "history_tiles_sha256": prio.file_sha256(path),
        "amw_files": summary["files"],
        "amw_fingerprints": summary["fingerprints"],
        "amw_locations": summary["amw_locations"],
        "amw_matched_to_our_chips": summary["matched_to_our_chips"],
    }
    return table, {"per_subregion": per_sub, "source": source}


def report(table: pd.DataFrame, extra: dict, rings: int) -> str:
    n1 = int((table["priority"] == 1).sum())
    lines = [f"rings       {rings} (priority 1 = AMW tiles"
             + ("" if rings == 0 else f" + {rings} ring{'s' * (rings > 1)}") + ")",
             f"tiles       {len(table):,}: priority 1 {n1:,} ({n1 / len(table):.1%}), "
             f"priority 2 {len(table) - n1:,}"]
    for k in range(int(table["ring"].max()) + 1):
        lines.append(f"  ring {k}{'+' if k == table['ring'].max() else ' '}    "
                     f"{int((table['ring'] == k).sum()):>8,} tiles")
    lines.append("per subregion (tasks score these many tiles in each pass):")
    for n, c in extra["per_subregion"].items():
        lines.append(f"  {n}: priority 1 {c.get('1', 0):,}, priority 2 {c.get('2', 0):,}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a run's priority list (once).")
    ap.add_argument("--run", required=True, help="the run folder made by plan.py")
    ap.add_argument("--rings", type=int, help="rings of neighbouring tiles in priority 1 (0-3)")
    ap.add_argument("--dry-run", action="store_true", help="show the counts, write nothing")
    ap.add_argument("--show", action="store_true", help="describe the run's existing list")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not (run_dir / "run" / "config.yaml").is_file():
        raise SystemExit(f"PRIORITY ERROR: {run_dir} is not a planned run")
    try:
        cfg = load_config(run_dir / "run" / "config.yaml")
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")

    if args.show:
        try:
            current = prio.load(run_dir)
        except prio.PriorityError as err:
            raise SystemExit(f"PRIORITY ERROR: {err}")
        if current is None:
            raise SystemExit(f"{run_dir} has no priority list yet")
        print(prio.describe(current))
        r = current.record
        print(f"made        {r['made_utc']} with code {str(r['git_commit'])[:10]}"
              + (" (uncommitted changes!)" if r.get("git_uncommitted_files") else ""))
        print(f"AMW files   {', '.join(r['source']['amw_files'])}")
        users = used_by(run_dir)
        print(f"used by     {len(users)} task folder(s)")
        raise SystemExit(0)

    if args.rings is None:
        raise SystemExit("PRIORITY ERROR: give --rings (or --show)")
    table, extra = build(run_dir, args.rings)
    print(report(table, extra, args.rings))
    if args.dry_run:
        print("\n--dry-run: nothing written")
        raise SystemExit(0)

    out = prio.folder(run_dir)
    if prio.exists(run_dir):
        try:
            current = prio.load(run_dir)
        except prio.PriorityError as err:
            raise SystemExit(f"PRIORITY ERROR: {err}")
        same = (current.rings == args.rings
                and current.of == dict(zip(table["tile"].tolist(), table["priority"].tolist())))
        if same:
            print(f"\nthe run already has exactly this list ({current.tag}); nothing written")
            raise SystemExit(0)
        users = used_by(run_dir)
        raise SystemExit(
            f"\nPRIORITY ERROR: the run already has a different list ({prio.describe(current)}). "
            + (f"{len(users)} task folder(s) have used it, so it can no longer change; "
               f"score the rest with --priority 2 instead." if users else
               f"No task has used it yet: to replace it, move {out} aside by hand and run "
               f"this again."))

    env = environment(cfg)
    record = {"made_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rings": args.rings,
              "tiles": {"1": int((table["priority"] == 1).sum()), "2": int((table["priority"] == 2).sum())},
              **extra, **{k: env[k] for k in ("host", "git_commit", "git_uncommitted_files")}}
    work = out.with_name(out.name + ".writing")
    work.mkdir(parents=True, exist_ok=True)
    table.to_csv(work / prio.TILES, index=False)
    record["tiles_sha256"] = prio.file_sha256(work / prio.TILES)
    (work / prio.RECORD).write_text(json.dumps(record, indent=2) + "\n")
    os.replace(work, out)                  # the folder appears only when complete
    print(f"\nwritten -> {out} (list {record['tiles_sha256'][:12]})")
    if env["git_uncommitted_files"]:
        print(f"WARNING     {len(env['git_uncommitted_files'])} file(s) differ from the last "
              f"commit (see record.json); commit before a real run")
