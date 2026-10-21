"""Variant model files of the switchable finals (`src.subjective_randomness.ablation.finals`).

A variant is a self-contained PyMC model file: a final's switchable source,
then a footer that builds it with some mechanisms on. Its name lists the
mechanisms removed (``r1_full``, ``r1_minus_pal_var``), so a name is a model.

A mechanism set is valid when every mechanism on has one of its ``REQUIRES``
on and each ``KEEP_ONE_OF`` group keeps a member. The removals open to a
model are the mechanisms whose removal leaves a valid set: person-level
variation goes before the parameter it varies (personal -> shared -> off).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Tuple

from src.subjective_randomness.ablation.finals import run1, run2, run3

FINALS: Dict[str, ModuleType] = {"run1": run1, "run2": run2, "run3": run3}
PREFIX = {"run1": "r1", "run2": "r2", "run3": "r3"}


def spec(final: str) -> ModuleType:
    if final not in FINALS:
        raise ValueError(f"unknown final {final!r}; known: {list(FINALS)}")
    return FINALS[final]


def canonical(final: str, mechanisms: Iterable[str]) -> Tuple[str, ...]:
    """Mechanisms in the final's ``MECHANISMS`` order (raises on an unknown one)."""
    order = spec(final).MECHANISMS
    ms = set(mechanisms)
    unknown = sorted(ms - set(order))
    if unknown:
        raise ValueError(f"unknown mechanisms of {final}: {unknown}; known: {list(order)}")
    return tuple(m for m in order if m in ms)


def invalid_reason(final: str, mechanisms: Iterable[str]) -> Optional[str]:
    """Why the set is not a model (``None`` when it is)."""
    s = spec(final)
    on = set(canonical(final, mechanisms))
    for m in sorted(on):
        needs = s.REQUIRES.get(m)
        if needs and not on & set(needs):
            return f"{m} needs one of {list(needs)}"
    for group in s.KEEP_ONE_OF:
        if not on & set(group):
            return f"one of {list(group)} must stay"
    return None


def removals(final: str, mechanisms: Iterable[str]) -> List[str]:
    """The mechanisms that can be removed from the set, in canonical order."""
    on = canonical(final, mechanisms)
    return [m for m in on if invalid_reason(final, set(on) - {m}) is None]


def variant_name(final: str, mechanisms: Iterable[str]) -> str:
    s = spec(final)
    on = set(canonical(final, mechanisms))
    off = [m for m in s.MECHANISMS if m not in on]
    return f"{PREFIX[final]}_" + ("minus_" + "_".join(s.SHORT[m] for m in off) if off else "full")


def variant_source(final: str, mechanisms: Iterable[str]) -> str:
    on = canonical(final, mechanisms)
    reason = invalid_reason(final, on)
    if reason:
        raise ValueError(f"{final} with {list(on)} is not a model: {reason}")
    source = Path(spec(final).__file__).read_text(encoding="utf-8")
    return (source + "\n\n# --- this variant (src.subjective_randomness.ablation.variants) ---\n"
            f"MECHANISMS_ON = {on!r}\nmodel = build(MECHANISMS_ON)\n")


def write_variant(final: str, mechanisms: Iterable[str], out_dir: Path) -> Tuple[str, Path]:
    """Write (or reuse) the variant's model file; returns its name and path."""
    name = variant_name(final, mechanisms)
    text = variant_source(final, mechanisms)
    path = Path(out_dir) / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} holds another version of {name} (sha "
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}); use a fresh directory")
        return name, path
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return name, path


def size(final: str, mechanisms: Iterable[str]) -> Dict[str, int]:
    """Free population-level scalars and person-level vectors (one entry per participant slot)."""
    import numpy as np

    model = spec(final).build(canonical(final, mechanisms))
    point = model.initial_point()
    sizes = [int(np.prod(point[model.rvs_to_values[rv].name].shape)) for rv in model.free_RVs]
    slots = getattr(spec(final), "MAX_PARTICIPANTS", None) or spec(final).N_SLOTS
    if any(n not in (1, slots) for n in sizes):
        raise ValueError(f"{final}: a free variable is neither a scalar nor one entry per slot ({sizes})")
    return {"population": sizes.count(1), "person_vectors": sizes.count(slots)}
