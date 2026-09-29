"""Pure-Python twins of the subjective-randomness PyMC model families.

Each module here implements the same likelihood as its PyMC adapter in
``../pymc_model_families/``, as a plain callable with a ``DEFAULT_PARAMS``
dict. Recovery uses the twins to generate ground-truth data at fixed
parameters and to define the no-learning baseline, and the test suite uses
them to check each PyMC adapter against an independent implementation.

Which models are *active* is not decided here: the registry manifest
(``pymc_model_families/models_manifest.yaml``) is the single source of truth,
and the outer loop's live seed pool mirrors it. Adding a module here does not
activate it; adding it to the registry manifest does.
"""

from . import falk_konold_dp
from . import finite_experience_occurrence
from . import local_representativeness
from . import motif_stack

__all__ = [
    "falk_konold_dp",
    "finite_experience_occurrence",
    "local_representativeness",
    "motif_stack",
]
