from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from periods import FREQUENCIES, Period, period_range

PIPELINE_DIR = Path(__file__).resolve().parent     # .../panel_pipeline
REPO = PIPELINE_DIR.parent                         # the repository root
DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"


@dataclass(frozen=True)
class Region:
    subregion_files: dict[int, Path]     # 1 -> .../Amazon_ACA_1.geojson, ...
    andes_boundary: Path

@dataclass(frozen=True)
class Imagery:
    collection: str
    tilesize: int
    pad: int
    clear_threshold: float
    max_workers: int


@dataclass(frozen=True)
class Inference:
    model_path: Path
    stride_ratio: int
    tries: int
    batch_tiles: int


@dataclass(frozen=True)
class DualThreshold:
    t_main: float
    t_iso: float


@dataclass(frozen=True)
class Thresholds:
    raw: float
    andes: float
    loose: DualThreshold
    stringent: DualThreshold
    neighbours_k: int
    isolation_km: float


@dataclass(frozen=True)
class Persistence:
    k: int
    window_months: int


@dataclass(frozen=True)
class Config:
    source: Path                          # the config.yaml these came from
    run_name: str
    output_root: Path
    periods: dict[str, list[Period]]      # "monthly" -> 105 periods, ...
    region: Region
    imagery: Imagery
    inference: Inference
    thresholds: Thresholds
    persistence: Persistence


# Checker helpers for configuration

def _section(raw: dict, name: str, keys: set[str]) -> dict:
    """One section of the YAML, with exactly the expected keys."""
    if not isinstance(raw, dict) or name not in raw:
        raise ValueError(f"config: section '{name}' is missing")
    sec = raw[name]
    if not isinstance(sec, dict):
        raise ValueError(f"config: section '{name}' is empty or not a section")
    missing, unknown = keys - sec.keys(), sec.keys() - keys
    if missing or unknown:
        raise ValueError(f"config: section '{name}' is missing {sorted(missing)} "
                         f"and has unknown keys {sorted(unknown)} (a typo?)")
    return sec


def _int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config: {name} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"config: {name} must be at least {minimum}, got {value}")
    return value


def _share(name: str, value) -> float:
    """A number strictly between 0 and 1 (a probability or a fraction)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config: {name} must be a number, got {value!r}")
    if not 0 < value < 1:
        raise ValueError(f"config: {name} must be between 0 and 1, got {value}")
    return float(value)


def _file(name: str, relative: str) -> Path:
    path = REPO / relative
    if not path.is_file():
        raise ValueError(f"config: {name} points to a missing file: {path}")
    return path


# final function that automates the checking

def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)

    run = _section(raw, "run", {"name", "output_root"})
    if run["output_root"] in (None, "", "CHANGE_ME"):
        raise ValueError("config: set run.output_root to a real folder first")

    per = _section(raw, "periods", {"first_month", "last_month", "frequencies"})
    if not isinstance(per["frequencies"], list) or not per["frequencies"] \
            or len(set(per["frequencies"])) != len(per["frequencies"]):
        raise ValueError("config: periods.frequencies must be a list like "
                         "[monthly, quarterly], without repeats")
    for freq in per["frequencies"]:
        if freq not in FREQUENCIES:
            raise ValueError(f"config: unknown frequency {freq!r}; "
                             f"allowed: {FREQUENCIES}")
    periods = {freq: period_range(per["first_month"], per["last_month"], freq)
               for freq in per["frequencies"]}

    reg = _section(raw, "region", {"subregion_files", "subregions", "andes_boundary"})
    if not isinstance(reg["subregions"], list) or not reg["subregions"]:
        raise ValueError("config: region.subregions must be a list like [1, 2, 3]")
    region = Region(
        subregion_files={_int("region.subregions", n, 1):
                         _file(f"subregion {n}", reg["subregion_files"].format(n=n))
                         for n in reg["subregions"]},
        andes_boundary=_file("region.andes_boundary", reg["andes_boundary"]),
    )

    mod = _section(raw, "model", {"path"})
    img = _section(raw, "imagery", {"collection", "tilesize", "pad",
                                    "clear_threshold", "max_workers"})
    if img["collection"] != "S2L1C":
        raise ValueError("config: imagery.collection must be S2L1C; "
                         "the model was trained on its 13 bands")
    imagery = Imagery(
        collection=img["collection"],
        tilesize=_int("imagery.tilesize", img["tilesize"], 1),
        pad=_int("imagery.pad", img["pad"], 0),
        clear_threshold=_share("imagery.clear_threshold", img["clear_threshold"]),
        max_workers=_int("imagery.max_workers", img["max_workers"], 1),
    )

    inf = _section(raw, "inference", {"stride_ratio", "tries", "batch_tiles"})
    inference = Inference(
        model_path=_file("model.path", mod["path"]),
        stride_ratio=_int("inference.stride_ratio", inf["stride_ratio"], 1),
        tries=_int("inference.tries", inf["tries"], 1),
        batch_tiles=_int("inference.batch_tiles", inf["batch_tiles"], 1),
    )

    thr = _section(raw, "thresholds", {"raw", "andes", "loose", "stringent",
                                       "neighbours_k", "isolation_km"})
    duals = {}
    for name in ("loose", "stringent"):
        d = _section(thr, name, {"t_main", "t_iso"})
        t_main = _share(f"thresholds.{name}.t_main", d["t_main"])
        t_iso = _share(f"thresholds.{name}.t_iso", d["t_iso"])
        if t_iso < t_main:
            raise ValueError(f"config: thresholds.{name}: t_iso must be at "
                             f"least t_main (isolated chips need MORE evidence)")
        duals[name] = DualThreshold(t_main, t_iso)
    km = thr["isolation_km"]
    if isinstance(km, bool) or not isinstance(km, (int, float)) or km <= 0:
        raise ValueError("config: thresholds.isolation_km must be a positive number")
    thresholds = Thresholds(
        raw=_share("thresholds.raw", thr["raw"]),
        andes=_share("thresholds.andes", thr["andes"]),
        loose=duals["loose"],
        stringent=duals["stringent"],
        neighbours_k=_int("thresholds.neighbours_k", thr["neighbours_k"], 1),
        isolation_km=float(km),
    )

    pst = _section(raw, "persistence", {"k", "window_months"})
    persistence = Persistence(
        k=_int("persistence.k", pst["k"], 1),
        window_months=_int("persistence.window_months", pst["window_months"], 1),
    )
    if "quarterly" in periods and persistence.window_months % 3:
        raise ValueError("config: persistence.window_months must be a multiple "
                         "of 3, so the window covers whole quarters")
    for freq in periods:
        step = 1 if freq == "monthly" else 3
        if persistence.k > persistence.window_months // step:
            raise ValueError(f"config: persistence.k={persistence.k} can never be "
                             f"reached: a {persistence.window_months}-month window "
                             f"holds only {persistence.window_months // step} "
                             f"{freq} periods")

    return Config(
        source=path,
        run_name=str(run["name"]),
        output_root=Path(run["output_root"]).expanduser(),
        periods=periods,
        region=region,
        imagery=imagery,
        inference=inference,
        thresholds=thresholds,
        persistence=persistence,
    )

def describe(cfg: Config) -> str:
    """A readable summary of the settings, for a person to check."""
    t, p = cfg.thresholds, cfg.persistence
    lines = [
        f"config       {cfg.source}",
        f"run          {cfg.run_name}  ->  {cfg.output_root}",
    ]
    for freq, periods in cfg.periods.items():
        first, last = periods[0], periods[-1]
        lines.append(f"{freq:<12} {len(periods)} periods: {first.tag} "
                     f"(from {first.start}) to {last.tag} (until {last.end}, excluded)")
    lines += [
        f"subregions   {sorted(cfg.region.subregion_files)}",
        f"andes        {cfg.region.andes_boundary.name}",
        f"model        {cfg.inference.model_path.name}",
        f"imagery      {cfg.imagery.collection}, tile {cfg.imagery.tilesize}"
        f"+{cfg.imagery.pad} px, clear >= {cfg.imagery.clear_threshold:g}",
        f"thresholds   amazon > {t.raw:g}, andes > {t.andes:g}, "
        f"loose {t.loose.t_main:g}/{t.loose.t_iso:g}, "
        f"stringent {t.stringent.t_main:g}/{t.stringent.t_iso:g}, "
        f"isolated = {t.neighbours_k}th neighbour beyond {t.isolation_km:g} km",
        f"persistence  {p.k} detections within {p.window_months} months",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check config.yaml and print what it sets.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to a config.yaml (default: the one next to this file)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")

    print(describe(cfg))
    print("\nconfig OK")