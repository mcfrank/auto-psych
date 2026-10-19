"""Helpers shared by the recovery test modules.

* ``CannedPredictionFit`` — a stand-in for a fitted PyMC model. Recovery tests
  monkeypatch ``fit_model`` so no MCMC runs, and the holdout path only ever
  asks a fit to predict on the held-out stimuli.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class CannedPredictionFit:
    """A fitted model that predicts a spread of p_left, for the holdout path.

    The predictions vary across stimuli (a constant would make every Pearson
    correlation undefined) and are identical for every model, so a test that
    recovers the ground truth reads r == 1.0 exactly.
    """

    model = None

    def predict_p_left(self, stim_data, var_name="p_left"):
        return np.linspace(0.1, 0.9, stim_data["n"])

    def loo_diagnostics(self):
        return SimpleNamespace(elpd_loo=-1.0, unreliable=False)

    def convergence_problems(self):
        return []

