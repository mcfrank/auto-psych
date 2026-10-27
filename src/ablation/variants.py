"""Variant model files of an ablation's switchable finals.

A switchable final is a source module defining ``FINAL`` (the model it
rewrites), ``MECHANISMS``, ``REQUIRES`` (mechanism -> the mechanisms it needs,
any one of them), ``KEEP_ONE_OF`` (groups of which a model keeps at least
one), ``SHORT`` (mechanism -> its short name in variant names), ``NEUTRAL``,
the final's data hook and ``build(on) -> pm.Model``.

A variant is a self-contained PyMC model file: a final's switchable source,
then a footer that builds it with some mechanisms on. Its name lists the
mechanisms removed (``r1_full``, ``r1_minus_pal_var``), so a name is a model.

A mechanism set is valid when every mechanism on has one of its ``REQUIRES``
on and each ``KEEP_ONE_OF`` group keeps a member. The removals open to a
model are the mechanisms whose removal leaves a valid set (for person-level
variation: personal -> shared -> off).
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Tuple


class Variants:
    """The variants of one ablation's finals.

    ``finals`` maps each final's key to its switchable source module, named as
    a string: the sources import pymc, and so arviz, so they are imported only
    when first asked for (a fit child re-imports the script it was started
    from and refuses to fit when arviz is already loaded). ``prefix`` maps
    each final to its variant names' prefix; ``module`` is named in every
    variant file's footer.
    """

    def __init__(self, finals: Dict[str, str], prefix: Dict[str, str], module: str):
        self.finals = finals
        self.prefix = prefix
        self.module = module

    def spec(self, final: str) -> ModuleType:
        if final not in self.finals:
            raise ValueError(f"unknown final {final!r}; known: {list(self.finals)}")
        return importlib.import_module(self.finals[final])

    def canonical(self, final: str, mechanisms: Iterable[str]) -> Tuple[str, ...]:
        """Mechanisms in the final's ``MECHANISMS`` order (raises on an unknown one)."""
        order = self.spec(final).MECHANISMS
        ms = set(mechanisms)
        unknown = sorted(ms - set(order))
        if unknown:
            raise ValueError(f"unknown mechanisms of {final}: {unknown}; known: {list(order)}")
        return tuple(m for m in order if m in ms)

    def invalid_reason(self, final: str, mechanisms: Iterable[str]) -> Optional[str]:
        """Why the set is not a model (``None`` when it is)."""
        s = self.spec(final)
        on = set(self.canonical(final, mechanisms))
        for m in sorted(on):
            needs = s.REQUIRES.get(m)
            if needs and not on & set(needs):
                return f"{m} needs one of {list(needs)}"
        for group in s.KEEP_ONE_OF:
            if not on & set(group):
                return f"one of {list(group)} must stay"
        return None

    def removals(self, final: str, mechanisms: Iterable[str]) -> List[str]:
        """The mechanisms that can be removed from the set, in canonical order."""
        on = self.canonical(final, mechanisms)
        return [m for m in on if self.invalid_reason(final, set(on) - {m}) is None]

    def variant_name(self, final: str, mechanisms: Iterable[str]) -> str:
        s = self.spec(final)
        on = set(self.canonical(final, mechanisms))
        off = [m for m in s.MECHANISMS if m not in on]
        return f"{self.prefix[final]}_" + ("minus_" + "_".join(s.SHORT[m] for m in off) if off else "full")

    def variant_source(self, final: str, mechanisms: Iterable[str]) -> str:
        on = self.canonical(final, mechanisms)
        reason = self.invalid_reason(final, on)
        if reason:
            raise ValueError(f"{final} with {list(on)} is not a model: {reason}")
        source = Path(self.spec(final).__file__).read_text(encoding="utf-8")
        return (source + f"\n\n# --- this variant ({self.module}) ---\n"
                f"MECHANISMS_ON = {on!r}\nmodel = build(MECHANISMS_ON)\n")

    def write_variant(self, final: str, mechanisms: Iterable[str], out_dir: Path) -> Tuple[str, Path]:
        """Write (or reuse) the variant's model file; returns its name and path."""
        name = self.variant_name(final, mechanisms)
        text = self.variant_source(final, mechanisms)
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

    def size(self, final: str, mechanisms: Iterable[str]) -> Dict[str, int]:
        """Free population-level scalars and person-level vectors (one entry per
        participant slot; a final without slots has no person-level vectors)."""
        import numpy as np

        s = self.spec(final)
        model = s.build(self.canonical(final, mechanisms))
        point = model.initial_point()
        sizes = [int(np.prod(point[model.rvs_to_values[rv].name].shape)) for rv in model.free_RVs]
        slots = getattr(s, "MAX_PARTICIPANTS", None) or getattr(s, "N_SLOTS", None)
        if any(n not in (1, slots) for n in sizes):
            raise ValueError(f"{final}: a free variable is neither a scalar nor one entry per slot ({sizes})")
        return {"population": sizes.count(1), "person_vectors": sizes.count(slots)}
