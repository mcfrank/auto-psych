"""The open chains' starting theories and the bar, ported from the closed campaign (PLAN.md,
"Seeds": each chain starts from its closed counterpart's seeds, *not* its finals; the bar is
again the 5 starting models plus the 12 promoted seeds).

    uv run python -m src.rsa_open.seeds            # writes data/rsa_open/seeds/{bar,chain_0,...}

Every theory is `src.rsa_open.port.port_file` of the closed file (hand ports included), with the
closed manifest's rationale. A name in both the starting models and the promoted seeds (the
references, rsa_l1 and rsa_l2) is one theory: its two originals must be the same file, else the
build raises. The output is deterministic; rebuilding over it rewrites it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import tyro
import yaml
from pyprojroot import here

from src.models.model_manifest import read_manifest_entries
from src.rsa.outer.run import STARTING_MODELS
from src.rsa_open.port import port_file

PROMOTED = here() / "data/rsa/live_seeds/models"
CHAINS = here() / "data/rsa/live_seeds/chains"
OUT = here() / "data/rsa_open/seeds"


def _entries(folder: Path) -> List[Tuple[str, Path, str]]:
    return [(e["name"], Path(folder) / f"{e['name']}.py", e.get("rationale", "")) for e in read_manifest_entries(folder)]


def _write(dst: Path, entries: List[Tuple[str, Path, str]]) -> Path:
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, path, _ in entries:
        (tmp / f"{name}.py").write_text(port_file(path))
    (tmp / "models_manifest.yaml").write_text(yaml.safe_dump(
        {"models": [{"name": n, "rationale": r} for n, _, r in entries]}, sort_keys=False, width=100))
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return dst


def bar_entries(starting: Path = STARTING_MODELS, promoted: Path = PROMOTED) -> List[Tuple[str, Path, str]]:
    out: Dict[str, Tuple[str, Path, str]] = {}
    for name, path, rationale in _entries(starting) + _entries(promoted):
        if name in out:
            if out[name][1].read_bytes() != path.read_bytes():
                raise ValueError(f"{name}: the starting model {out[name][1]} and the promoted seed {path} differ")
            continue
        out[name] = (name, path, rationale)
    return list(out.values())


def build(out: Path = OUT, starting: Path = STARTING_MODELS, promoted: Path = PROMOTED, chains: Path = CHAINS) -> dict:
    bar = bar_entries(starting, promoted)
    _write(Path(out) / "bar", bar)
    split = json.loads((Path(chains) / "chains.json").read_text())
    record = {"bar": [n for n, _, _ in bar], "chains": {}}
    for c in split["chains"]:
        name = f"chain_{c['chain']}"
        entries = _entries(Path(chains) / name)
        if sorted(n for n, _, _ in entries) != sorted(c["models"]):
            raise ValueError(f"{chains / name}'s manifest is not chains.json's chain {c['chain']}")
        _write(Path(out) / name, entries)
        record["chains"][name] = [n for n, _, _ in entries]
    record["source"] = {"starting": str(Path(starting).relative_to(here())),
                        "promoted": str(Path(promoted).relative_to(here())),
                        "chains": str(Path(chains).relative_to(here()))}
    (Path(out) / "seeds.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


@dataclass
class Args:
    out: Path = OUT


if __name__ == "__main__":
    print(json.dumps(build(tyro.cli(Args).out), indent=1))
