"""Does one normal Earth Engine request get through from Goethe with TCP keepalive on?

    source cluster/goethe_env.sh
    python cluster/check_keepalive.py --run /work/<group>/<user>/runs/<run> [--period 2025Q1]

Background (25-26 Oct 2026): from Goethe, login and compute nodes alike, a request that
Google answers within seconds arrives, but one that needs 30 s or more of server-side
work before the first byte never does: the connection stays silent until our 300 s
timeout. A firewall or address translator that drops connections idle for some tens
of seconds would behave exactly like that. TCP keepalive makes the operating system
send a tiny probe every few seconds on an otherwise idle connection, which such a box
counts as traffic.

The script sends detect.py's own request (the `panel` recipe, one tile) through a
connection with keepalive switched on, and prints how long the answer took. One
attempt, no retries: either a line with seconds and MB within about two minutes, or a
TimeoutError after 300 s. Nothing is written.

Outcome (26 Oct 2026): not needed. The same request timed out from Ruby's Mac too,
so the network was never the cause; it was an ee.Image.constant without a footprint
in detect.quality_bands (fixed the same evening). Kept as a diagnostic in case a
network question comes up again.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import ee
import pandas as pd
import requests
from descarteslabs.geo import DLTile
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "panel_pipeline"))          # detect.py
sys.path.insert(0, str(ROOT / "panel_pipeline" / "lib"))  # config, periods
sys.path.insert(0, str(ROOT / "core"))                    # AMW's gee.py

import detect                                              # noqa: E402
from config import load_config                             # noqa: E402
from periods import Period                                 # noqa: E402
import gee                                                 # noqa: E402

TIMEOUT_S = 300          # as detect.py (REQUEST_TIMEOUT_S)
KEEPIDLE = getattr(socket, "TCP_KEEPIDLE", None) or socket.TCP_KEEPALIVE   # Linux name, macOS name
KEEPALIVE = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, KEEPIDLE, 10),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),      # first probe after 10 s idle, then every 10 s
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 30),        # give up after 30 unanswered probes (5 min)
]


class KeepAliveAdapter(HTTPAdapter):
    """requests' connection pool, with keepalive on every socket it opens."""

    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = HTTPConnection.default_socket_options + KEEPALIVE
        super().init_poolmanager(*args, **kwargs)


def keepalive_transport():
    """The Earth Engine client's HTTP transport, on a session whose sockets keep alive."""
    session = requests.Session()
    session.mount("https://", KeepAliveAdapter())
    return ee._cloud_api_utils._Http(session, timeout=TIMEOUT_S)   # what Initialize builds itself


_initialize = ee.Initialize


def initialize_with_keepalive(*args, **kwargs):
    """ee.Initialize as AMW's gee.py calls it, plus our transport."""
    kwargs["http_transport"] = keepalive_transport()
    return _initialize(*args, **kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One Earth Engine request with TCP keepalive.")
    ap.add_argument("--run", required=True, help="a planned run folder (plan/tiles.csv)")
    ap.add_argument("--period", default="2025Q1")
    ap.add_argument("--start", type=int, default=30000, help="row in plan/tiles.csv")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    cfg = load_config(run_dir / "run" / "config.yaml")
    period = Period.parse(args.period)
    key = pd.read_csv(run_dir / "plan" / "tiles.csv")["tile_key"][args.start]
    tile = DLTile.from_key(key)

    ee.Initialize = initialize_with_keepalive      # gee.py looks the name up when it logs in
    gee.EE_PROJECT = cfg.ee_project
    gee._ensure_earth_engine_initialized()
    ee.data.setMaxRetries(0)                       # one attempt, so a failure shows in 300 s
    print(f"logged in with keepalive; {period.tag}, tile {key}, timeout {TIMEOUT_S} s", flush=True)

    panel = detect.extractor_for(cfg, period, None)
    started = time.perf_counter()
    pixels = panel.get_tile_data(tile)
    seconds = time.perf_counter() - started
    print(f"panel    {key:<26} {seconds:6.1f} s, {pixels.nbytes / 1e6:5.1f} MB of pixels   "
          f"<- it got through", flush=True)
