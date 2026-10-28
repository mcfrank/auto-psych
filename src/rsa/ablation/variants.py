"""Variant model files of the RSA ablation's switchable final (`src.ablation.variants`, for memo models).

A variant is a self-contained memo model file: the switchable source (``spec.SOURCE``) and a
footer that builds it with some mechanisms on, ``PARAMS, choice_probs = build(MECHANISMS_ON)``.
Naming, validity (``REQUIRES``) and removals are the sr-oct26 ablation's, unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from src.ablation.variants import Variants


class RSAVariants(Variants):
    def variant_source(self, final: str, mechanisms: Iterable[str]) -> str:
        on = self.canonical(final, mechanisms)
        reason = self.invalid_reason(final, on)
        if reason:
            raise ValueError(f"{final} with {list(on)} is not a model: {reason}")
        source = Path(self.spec(final).SOURCE).read_text(encoding="utf-8")
        return (source + f"\n\n# --- this variant ({self.module}) ---\n"
                f"MECHANISMS_ON = {on!r}\nPARAMS, choice_probs = build(MECHANISMS_ON)\n")

    def size(self, final: str, mechanisms: Iterable[str]) -> Dict[str, int]:
        """Free parameters (all scalars: the RSA models have no person-level vectors)."""
        s = self.spec(final)
        on = self.canonical(final, mechanisms)
        return {"population": len(s.CORE) + sum(len(s.PARAMETERS[m]) for m in on), "person_vectors": 0}


VARIANTS = RSAVariants(finals={"c0": "src.rsa.ablation.spec_c0"}, prefix={"c0": "c0"},
                       module="src.rsa.ablation.variants")
