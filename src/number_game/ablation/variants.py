"""Variant model files of the switchable number-game final (`src.number_game.ablation.final`).

A variant is the final's switchable source plus a footer that builds it with
some mechanisms on; its name lists the mechanisms removed (``ng_full``,
``ng_minus_par_even``). The functions are those of
`src.ablation.variants.Variants`, for this one final, whose key is ``ng``.
"""

from __future__ import annotations

from src.ablation.variants import Variants

# The final imports pymc, and so arviz, so it is imported only when first asked
# for: a fit child re-imports the script it was started from (`eliminate.py`)
# and refuses to fit when arviz is already loaded.
VARIANTS = Variants({"ng": "src.number_game.ablation.final"}, {"ng": "ng"},
                    module="src.number_game.ablation.variants")

spec = VARIANTS.spec
canonical = VARIANTS.canonical
invalid_reason = VARIANTS.invalid_reason
removals = VARIANTS.removals
variant_name = VARIANTS.variant_name
variant_source = VARIANTS.variant_source
write_variant = VARIANTS.write_variant
size = VARIANTS.size
