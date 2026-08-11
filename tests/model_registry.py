"""The active seed-model set, read from the registry manifest.

``pymc_model_families/models_manifest.yaml`` is the single source of truth for
which models are active (the 2026-08 literature-faithful consolidation). Nine
test sites had re-typed the four names by hand, so consolidating the set again
would mean editing all nine — and each one that was missed would fail for a
reason that looked like a bug in the code under test.

The one deliberate literal pin of *which* models these are lives in
``test_literature_faithful_pymc.py``; it is what makes the manifest itself
reviewable. Everything else derives the set from here.
"""

from __future__ import annotations

from src.models.model_manifest import read_manifest_names
from src.subjective_randomness.pymc_model_families import REGISTRY_DIR

# The superseded families each faithful model replaced. Their modules stay
# importable (archival refits, and holdout runs use one as an out-of-pool
# ground truth), so "not enumerating the package directory" is a property worth
# asserting rather than assuming. motif_hmm joined this set on 2026-08-10 when
# the four-motif stack automaton (motif_stack) superseded it as the faithful
# Griffiths et al. (2018) anchor.
SUPERSEDED_MODEL_NAMES = frozenset(
    {
        "encoding_compressibility",
        "bayesian_diagnosticity",
        "window_typicality",
        "prototype_similarity",
        "motif_hmm",
    }
)


def faithful_model_names() -> list[str]:
    """Active seed-model names, in manifest order."""
    return read_manifest_names(REGISTRY_DIR)


FAITHFUL_MODEL_NAMES = frozenset(faithful_model_names())
