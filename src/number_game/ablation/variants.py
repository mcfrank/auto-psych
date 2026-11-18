"""Variant model files of the switchable number-game finals.

``ng``: run 2's ``neighbourhood_doubt_three_example_memory`` (`src.number_game.ablation.final`),
the best winner on all runs pooled. ``ew``: run 3's ``exception_weighted_rule_rejection``
(`src.number_game.ablation.final_ew`), the best winner on the items no run drew (Figure 2).

A variant is a final's switchable source plus a footer that builds it with
some mechanisms on; its name lists the mechanisms removed (``ng_full``,
``ew_minus_exc_pre``). The functions are those of `src.ablation.variants.Variants`.
"""

from __future__ import annotations

from src.ablation.variants import Variants

# The final imports pymc, and so arviz, so it is imported only when first asked
# for: a fit child re-imports the script it was started from (`eliminate.py`)
# and refuses to fit when arviz is already loaded.
VARIANTS = Variants({"ng": "src.number_game.ablation.final", "ew": "src.number_game.ablation.final_ew"},
                    {"ng": "ng", "ew": "ew"},
                    module="src.number_game.ablation.variants")

spec = VARIANTS.spec
canonical = VARIANTS.canonical
invalid_reason = VARIANTS.invalid_reason
removals = VARIANTS.removals
variant_name = VARIANTS.variant_name
variant_source = VARIANTS.variant_source
write_variant = VARIANTS.write_variant
size = VARIANTS.size
