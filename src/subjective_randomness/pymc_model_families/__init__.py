"""PyMC adapters for the canonical subjective-randomness model families."""

from pathlib import Path

# The recovery registry. Its ``models_manifest.yaml`` is the single source of
# truth for which models are active: the outer loop's live seed pool mirrors
# it, and ``stimulus_design.default_model_family_names`` reads it directly.
REGISTRY_DIR = Path(__file__).resolve().parent

__all__ = ["REGISTRY_DIR"]
