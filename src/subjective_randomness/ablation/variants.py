"""Variant model files of the switchable finals (`src.subjective_randomness.ablation.finals`).

A variant is a self-contained PyMC model file: a final's switchable source,
then a footer that builds it with some mechanisms on. Its name lists the
mechanisms removed (``r1_full``, ``r1_minus_pal_var``), so a name is a model.

A mechanism set is valid when every mechanism on has one of its ``REQUIRES``
on and each ``KEEP_ONE_OF`` group keeps a member. The removals open to a
model are the mechanisms whose removal leaves a valid set: person-level
variation goes before the parameter it varies (personal -> shared -> off).

The functions are those of `src.ablation.variants.Variants`, for these finals.
"""

from __future__ import annotations

from typing import Dict

from src.ablation.variants import Variants

# The finals import pymc, and so arviz, so they are imported only when first
# asked for: a fit child re-imports the script it was started from
# (`eliminate.py`) and refuses to fit when arviz is already loaded.
FINALS: Dict[str, str] = {f"run{r}": f"src.subjective_randomness.ablation.finals.run{r}" for r in (1, 2, 3)}
PREFIX = {"run1": "r1", "run2": "r2", "run3": "r3"}
VARIANTS = Variants(FINALS, PREFIX, module="src.subjective_randomness.ablation.variants")

spec = VARIANTS.spec
canonical = VARIANTS.canonical
invalid_reason = VARIANTS.invalid_reason
removals = VARIANTS.removals
variant_name = VARIANTS.variant_name
variant_source = VARIANTS.variant_source
write_variant = VARIANTS.write_variant
size = VARIANTS.size
