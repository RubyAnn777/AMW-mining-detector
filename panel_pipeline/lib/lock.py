"""One writer per folder: a lock file that says who is writing, and keeps saying it.

detect.py takes the lock of a period's folder before it touches the folder and gives it
back when the period is done. Another copy of detect.py that finds the lock leaves the
folder alone. status.py reads the locks to show which tasks are running right now.

    <run>/detect/<frequency>/period=<tag>/subregion=<n>/_lock.json
        {"token": ..., "host": ..., "pid": ..., "job": ..., "started_utc": ..., "command": ...}

The file's modification time is the heartbeat: the holder touches it every minute. A
lock whose heartbeat stopped long ago belongs to a process that died without cleaning
up (killed, out of memory, a lost node), and the next process takes it over.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ADDITION (AMW has no locks: its VMs run one tmux session per job, started by hand).
LOCK_NAME = "_lock.json"      # starts with "_", so Parquet readers skip it like _status.json
HEARTBEAT_S = 60              # the holder touches the lock this often ...
STALE_AFTER_S = 15 * 60       # ... so a lock untouched this long has no living holder
JOB_VARIABLES = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID")   # common schedulers


class Locked(Exception):
    """The folder is locked by another process that is still alive."""


def _alive(pid: int) -> bool:
    """Whether a process with this number runs on this machine."""
    try:
        os.kill(pid, 0)              # signal 0 sends nothing; it only asks "do you exist?"
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                  # it exists, it just belongs to someone else
    return True


def read(folder: Path) -> dict | None:
    """The folder's lock, with its heartbeat age and whether it is stale; None if unlocked."""
    path = folder / LOCK_NAME
    try:
        age = time.time() - path.stat().st_mtime
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:     # created a moment ago, content not written yet
        info = {}
    # On the holder's own machine we can ask whether its process still runs; elsewhere
    # only the heartbeat tells.
    dead = (info.get("host") == platform.node() and "pid" in info
            and not _alive(info["pid"]))
    return info | {"heartbeat_age_s": round(age), "dead": dead,
                   "stale": dead or age > STALE_AFTER_S}


def who(info: dict) -> str:
    """A lock's holder, in words."""
    job = f", job {info['job']}" if info.get("job") else ""
    gone = ", no longer running" if info.get("dead") else ""
    return (f"{info.get('host', '?')} (process {info.get('pid', '?')}{job}{gone}), "
            f"since {info.get('started_utc', '?')}, "
            f"last heartbeat {info['heartbeat_age_s']} s ago")


class FolderLock:
    """Use as `with FolderLock(folder) as lock:`; raises Locked if someone else holds it."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / LOCK_NAME
        self.token = uuid.uuid4().hex            # a random name that only this holder knows
        self._stop = threading.Event()

    def __enter__(self) -> FolderLock:
        self.folder.mkdir(parents=True, exist_ok=True)
        for _ in range(3):                       # again after a lock vanished or went stale
            try:
                # O_CREAT | O_EXCL: create the file only if it does not exist yet, in one
                # step that the operating system (or a network file system) never splits.
                # Of two processes trying at once, exactly one succeeds.
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                held = read(self.folder)
                if held is None:                 # released between our two looks
                    continue
                if not held["stale"]:
                    raise Locked(f"{self.folder} is being written by {who(held)}")
                print(f"  taking over a stale lock in {self.folder}: {who(held)}",
                      flush=True)
                again = read(self.folder)        # remove it only if nobody beat us to it
                if again and again.get("token") == held.get("token"):
                    self.path.unlink(missing_ok=True)
                continue
            job = next((os.environ[v] for v in JOB_VARIABLES if v in os.environ), None)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"token": self.token, "host": platform.node(),
                               "pid": os.getpid(), "job": job,
                               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "command": " ".join(sys.argv)}, f, indent=2)
            except BaseException:                # stopped half-way: leave no empty lock
                self.path.unlink(missing_ok=True)
                raise
            threading.Thread(target=self._beat, daemon=True).start()
            return self
        raise Locked(f"{self.folder}: another process took the lock at the same moment")

    def _beat(self) -> None:
        """Touch the lock every HEARTBEAT_S seconds until released (a background thread)."""
        while not self._stop.wait(HEARTBEAT_S):
            try:
                os.utime(self.path)              # "still here": sets its time to now
            except FileNotFoundError:
                return                           # gone; check() stops the next write

    def check(self) -> None:
        """Stop unless this lock is still ours. Called before every write into the folder."""
        held = read(self.folder)
        if held is None or held.get("token") != self.token:
            raise SystemExit(f"LOCK ERROR: {self.folder} is no longer locked by this "
                             f"process{'' if held is None else ' but by ' + who(held)}; "
                             f"stopping without writing")

    def __exit__(self, *exc) -> None:
        self._stop.set()
        held = read(self.folder)
        if held is not None and held.get("token") == self.token:
            self.path.unlink(missing_ok=True)
