"""Helpers shared by the recovery test modules.

* ``CannedPredictionFit`` — a stand-in for a fitted PyMC model. Recovery tests
  monkeypatch ``fit_model`` so no MCMC runs, and the holdout path only ever
  asks a fit to predict on the held-out stimuli.
* ``p_left_model_family`` — ``p_left`` from a registry model's pure-Python
  twin, the independent implementation each PyMC adapter is checked against.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np


class CannedPredictionFit:
    """A fitted model that predicts a spread of p_left, for the holdout path.

    The predictions vary across stimuli (a constant would make every Pearson
    correlation undefined) and are identical for every model, so a test that
    recovers the ground truth reads r == 1.0 exactly.
    """

    model = None

    def predict_p_left(self, stim_data):
        return np.linspace(0.1, 0.9, stim_data["n"])

    def loo_diagnostics(self):
        return SimpleNamespace(elpd_loo=-1.0, unreliable=False)

    def convergence_problems(self):
        return []


def p_left_model_family(
    model_name: str,
    stimuli: Sequence[Mapping[str, str]],
    params: Mapping[str, float],
) -> np.ndarray:
    """``p_left`` per stimulus from the pure-Python model family ``model_name``.

    Fails loudly unless ``params`` names exactly the family's parameters.
    """
    module = importlib.import_module(
        f"src.subjective_randomness.model_families.{model_name}"
    )
    expected = set(module.DEFAULT_PARAMS)
    if set(params) != expected:
        missing = sorted(expected - set(params))
        extra = sorted(set(params) - expected)
        raise ValueError(
            f"Generating params must name exactly {model_name}'s parameters "
            f"{sorted(expected)}. Missing: {missing}. Unexpected: {extra}."
        )
    return np.array(
        [module.predict_left(stim, dict(params)) for stim in stimuli], dtype="float64"
    )
