"""PyMC adapters for the canonical subjective-randomness model families."""

from pathlib import Path

# The recovery registry. Its ``models_manifest.yaml`` is the single source of
# truth for which models are active: the outer loop's live seed pool mirrors
# it.
REGISTRY_DIR = Path(__file__).resolve().parent

__all__ = ["REGISTRY_DIR"]
