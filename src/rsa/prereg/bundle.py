"""The files registered on OSF with the pre-registration, checked and hashed.

    uv run python -m src.rsa.prereg.bundle --out <dir>

Collects what the registration freezes into ``<out>/prereg_final_osf/`` and a
zip of it: the plan (PREREG_FINAL.md, OSF_PREREG.md), the frozen predictions,
the finals' code, the design, the power records, the analysis code and the
launch config, with ``SHA256SUMS.txt``. It refuses an unfrozen plan: the
config must name the design's and the predictions' sha256, and every file must
match them. The PI uploads the zip to the OSF project and registers it
(docs/auto_rsa/OSF_PREREG.md); nothing here talks to OSF.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tyro
import yaml

from src.runtime.config import REPO_ROOT

CONFIG = Path("scripts/rsa/live/rsa_prereg.yaml")
DESIGN_DIR = Path("data/rsa/prereg/design")
PLAN_FILES = [
    Path("docs/auto_rsa/PREREG_FINAL.md"),
    Path("docs/auto_rsa/OSF_PREREG.md"),
    CONFIG,
    Path("data/rsa/live/finals/models_manifest.yaml"),
    Path("data/rsa/live/finals/boundary_avoidable_ambiguity_listener.py"),
    Path("data/rsa/live/finals/feature_surprisal_ensemble_listener.py"),
    Path("data/rsa/live/finals/competition_endpoint_binding_lapse_l2.py"),
]
DESIGN_FILES = [
    DESIGN_DIR / "design.json",
    DESIGN_DIR / "trial_lists.json",
    DESIGN_DIR / "prereg_design.json",
    DESIGN_DIR / "eig_half" / "eig.json",
    Path("data/rsa/prereg/power.json"),
    Path("data/rsa/prereg/design_power.json"),
]
CODE_DIR = Path("src/rsa/prereg")  # the analysis, power, design and run code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files(root: Path = REPO_ROOT) -> List[Tuple[Path, str]]:
    """Every registered file with its sha256, after checking the freeze."""
    cfg = yaml.safe_load((root / CONFIG).read_text())
    if not cfg.get("design_sha256"):
        raise ValueError(f"{CONFIG} has no design_sha256: the design is not frozen yet")
    design = json.loads((root / DESIGN_DIR / "prereg_design.json").read_text())
    if design["design_sha256"] != cfg["design_sha256"]:
        raise ValueError(f"{CONFIG} names design {cfg['design_sha256']}, {DESIGN_DIR} holds {design['design_sha256']}")
    if sha256(root / DESIGN_DIR / "trial_lists.json") != design["trial_lists_sha256"]:
        raise ValueError("trial_lists.json is not the design's recorded trial lists")
    if (cfg["participants"], cfg["displays"]) != (design["participants"], design["displays"]):
        raise ValueError(f"{CONFIG} recruits {cfg['participants']} for {cfg['displays']} displays; the design "
                         f"is {design['participants']} for {design['displays']}")
    preds = []
    for p in cfg["predictions"]:
        if not p.get("sha256") or sha256(root / p["path"]) != p["sha256"]:
            raise ValueError(f"{p['path']}: its sha256 is missing from {CONFIG} or does not match the file")
        preds.append(Path(p["path"]))
    plan = (root / "docs/auto_rsa/PREREG_FINAL.md").read_text()
    if cfg["design_sha256"] not in plan or any(p["sha256"] not in plan for p in cfg["predictions"]):
        raise ValueError("PREREG_FINAL.md §8 does not record the design's and the predictions' sha256")
    code = sorted(f.relative_to(root) for f in (root / CODE_DIR).glob("*.py"))
    files = PLAN_FILES + preds + DESIGN_FILES + code
    missing = [str(f) for f in files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered files: {missing}")
    return [(f, sha256(root / f)) for f in files]


@dataclass
class Args:
    out: Path
    """Where the bundle and its zip go (e.g. the scratchpad; not committed)."""


def main(args: Args) -> Path:
    files = frozen_files()
    dest = Path(args.out) / "prereg_final_osf"
    if dest.exists():
        shutil.rmtree(dest)
    for f, _ in files:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / f, dest / f)
    (dest / "SHA256SUMS.txt").write_text("".join(f"{h}  {f}\n" for f, h in files))
    archive = Path(shutil.make_archive(str(dest), "zip", dest.parent, dest.name))
    print(f"{len(files)} files -> {dest}\nzip: {archive} (sha256 {sha256(archive)})")
    return archive


if __name__ == "__main__":
    main(tyro.cli(Args))
