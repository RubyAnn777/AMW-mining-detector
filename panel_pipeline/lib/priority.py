"""Two-stage scoring: which tiles are priority 1 and which priority 2, once per run.

Priority 1 = every tile in which AMW has ever detected mining (any chip in any of AMW's
raw detection files), plus the rings of tiles around them. Priority 2 = every other tile.
Both are scored into the same task folders as a normal run, so a task that has had both
passes holds exactly what a single full pass would hold, and assembly, the checks and the
notebooks need no change.

    <run>/priority/tiles.csv     one row per tile of the run: tile_key, tile, ring, priority
    <run>/priority/record.json   how the list was made: rings, AMW files and their
                                 fingerprints, code version, counts, and tiles.csv's
                                 fingerprint

The list is made once (panel_pipeline/priority.py) and never changes afterwards, so every
task scored with --priority can say exactly which tiles it was given. load() refuses a
tiles.csv whose fingerprint no longer matches the record (edited by hand, or copied over
from another run), and detect.py writes the fingerprint into every _status.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FOLDER = "priority"
TILES = "tiles.csv"
RECORD = "record.json"
PRIORITIES = (1, 2)


def number_of_key(tile_key: str) -> int:
    """A tile's number (chip_id // 1000) from its key, '552:12:10.0:21:60:-100' (as grid.chip_ids)."""
    zone, col, row = (int(v) for v in tile_key.split(":")[3:])
    return (zone * 1000 + col + 500) * 10_000 + row + 5000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PriorityError(Exception):
    """The priority list is missing, changed, or does not fit the run."""


@dataclass(frozen=True)
class PriorityList:
    """A run's frozen priority list: tile number -> 1 or 2."""
    of: dict[int, int]
    rings: int
    sha256: str            # of tiles.csv, as record.json states and the file confirms
    record: dict

    @property
    def tag(self) -> str:
        """Short name for logs and _status.json: the first 12 characters of the fingerprint."""
        return self.sha256[:12]

    def priority(self, tile_number: int) -> int:
        try:
            return self.of[tile_number]
        except KeyError:
            raise PriorityError(f"tile {tile_number} is not in the priority list; "
                                f"was the list made for another run?") from None

    def select(self, tiles: list, which: int, number=lambda t: t) -> list:
        """The tiles of priority `which`, in their given order. `number` maps a tile to its number."""
        return [t for t in tiles if self.priority(number(t)) == which]

    def counts(self, planned: list[int], saved: set[int], failed: set[int] = frozenset()) -> dict:
        """Per priority: tiles planned in this task, saved, and failed (tile numbers given)."""
        out = {}
        for p in PRIORITIES:
            mine = [n for n in planned if self.priority(n) == p]
            out[str(p)] = {"planned": len(mine),
                           "saved": sum(n in saved for n in mine),
                           "failed": sum(n in failed for n in mine)}
        return out


def folder(run_dir: Path) -> Path:
    return Path(run_dir) / FOLDER


def exists(run_dir: Path) -> bool:
    return (folder(run_dir) / RECORD).is_file()


def load(run_dir: Path) -> PriorityList | None:
    """The run's priority list, checked against its record; None if the run has none."""
    rec_path, tiles_path = folder(run_dir) / RECORD, folder(run_dir) / TILES
    if not rec_path.is_file():
        return None
    record = json.loads(rec_path.read_text())
    if not tiles_path.is_file():
        raise PriorityError(f"{tiles_path} is missing, but {rec_path.name} says a list was made")
    actual = file_sha256(tiles_path)
    if actual != record["tiles_sha256"]:
        raise PriorityError(f"{tiles_path} has changed since it was made (fingerprint "
                            f"{actual[:12]}, recorded {record['tiles_sha256'][:12]}); a "
                            f"priority list is made once per run and never edited")
    with open(tiles_path, newline="") as f:
        of = {int(r["tile"]): int(r["priority"]) for r in csv.DictReader(f)}
    return PriorityList(of=of, rings=int(record["rings"]), sha256=actual, record=record)


def describe(p: PriorityList) -> str:
    """One line: what priority 1 is and how big both are."""
    n1 = sum(v == 1 for v in p.of.values())
    ring = "AMW tiles only" if p.rings == 0 else f"AMW tiles + {p.rings} ring{'s' * (p.rings > 1)}"
    return (f"priority 1 = {ring}: {n1:,} of {len(p.of):,} tiles ({n1 / len(p.of):.1%}); "
            f"priority 2 = the other {len(p.of) - n1:,} (list {p.tag})")
