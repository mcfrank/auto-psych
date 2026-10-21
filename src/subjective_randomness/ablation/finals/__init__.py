"""The three finals with switchable mechanisms, one source file each.

A source defines ``FINAL`` (the run's final it rewrites), ``MECHANISMS``,
``REQUIRES`` (mechanism -> the mechanisms it needs, any one of them),
``KEEP_ONE_OF`` (groups of which a model keeps at least one), ``NEUTRAL``
(mechanism -> the parameter values at which the model with it equals the
model without it; ``None`` when it is removed by structure), the final's
data hook, and ``build(on) -> pm.Model``. With every mechanism on, ``build``
is the final itself (tests/test_sr_ablation.py). A variant model file is the
source plus a footer that calls ``build`` (``src.subjective_randomness.ablation.variants``).
"""
