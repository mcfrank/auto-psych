"""Impossible-theory ground-truth generators for holdout recovery.

Each module here is a PyMC model (module-level ``model: pm.Model``) in the same
shape as the project seed models, but encoding a theory of subjective randomness
that humans could not plausibly use — e.g. "more heads => more random-looking".
They exist purely as ground-truth generators for the impossible-theory holdout
analysis (``scripts/subjective_randomness/impossible_holdout_recovery.py``).

This directory has NO ``models_manifest.yaml`` on purpose: the agentic loop is
seeded from the project's seed-model manifest, so these models can never enter
the agents' seed pool. Each model's only free parameters are ``beta`` (inverse
temperature) and ``side_bias``; the impossible structure is a deterministic
function of stimulus features with no free shape parameters, so the generating
params are exactly ``{beta, side_bias}``.
"""
