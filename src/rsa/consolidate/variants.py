"""Variant model files of the consolidated superset (`src.rsa.consolidate.superset`).

A variant is a self-contained model file: the superset's source, then a footer
that switches on its mechanisms. Its name lists them, so a name is a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Tuple

from src.rsa.consolidate.superset import MECHANISMS

SOURCE = Path(__file__).with_name("superset.py")
SHORT = {"depth2": "d2", "dup": "dup", "richness": "rich", "empty": "emp", "surprisal": "sur",
         "redundancy": "red", "aspect": "asp", "interior": "int", "avoidable": "avo", "competition": "comp",
         "confusion": "conf", "binding": "bind", "attention": "att"}


def canonical(mechanisms: Iterable[str]) -> Tuple[str, ...]:
    """Mechanisms in ``MECHANISMS`` order (raises on an unknown one)."""
    ms = set(mechanisms)
    unknown = sorted(ms - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")
    return tuple(m for m in MECHANISMS if m in ms)


def variant_name(mechanisms: Iterable[str]) -> str:
    ms = canonical(mechanisms)
    return "cons_" + ("_".join(SHORT[m] for m in ms) if ms else "core")


def write_variant(mechanisms: Iterable[str], out_dir: Path) -> Tuple[str, Path]:
    """Write (or reuse) the variant's model file; returns its name and path."""
    ms = canonical(mechanisms)
    name = variant_name(ms)
    text = (SOURCE.read_text() + "\n\n# --- this variant (src.rsa.consolidate.variants) ---\n"
            f"MECHANISMS_ON = {ms!r}\nPARAMS, choice_probs = make(MECHANISMS_ON)\n")
    path = Path(out_dir) / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        if path.exists():
            raise ValueError(f"{path} holds another version of {name} (sha "
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}); use a fresh directory")
        path.write_text(text)
    return name, path
