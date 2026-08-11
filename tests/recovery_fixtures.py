"""Stand-ins for a fitted PyMC model, shared by the recovery test modules.

Recovery tests monkeypatch ``fit_model`` so no MCMC runs, and every one of them
needs the same shape of object back. There are two shapes, because the two
recovery paths ask a fit for different things:

* ``CannedPosteriorFit`` — carries an ``idata`` with a fixed posterior. The
  ``recover``/``pipeline`` path reads parameter estimates out of it.
* ``CannedPredictionFit`` — answers ``predict_p_left``. The holdout path only
  ever asks a fit to predict on the held-out stimuli.

Each was duplicated verbatim in two test modules before it lived here.
"""

from __future__ import annotations

import numpy as np

# The canned posterior: two chains x two draws for each parameter the
# subjective-randomness families expose.
CANNED_POSTERIOR = {
    "theta_alt": [[0.60, 0.70], [0.65, 0.67]],
    "alt_weight": [[0.50, 0.55], [0.58, 0.57]],
    "beta": [[3.8, 4.2], [4.0, 4.1]],
    "side_bias": [[-0.1, 0.0], [0.1, 0.0]],
}


class FakeParam:
    """One posterior variable, exposing ``.values`` like an xarray DataArray."""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)


class FakeIdata:
    """An InferenceData stand-in: ``.posterior`` maps a name to a FakeParam."""

    def __init__(self, params):
        self.posterior = {name: FakeParam(values) for name, values in params.items()}


class CannedPosteriorFit:
    """A fitted model whose posterior is fixed, for the parameter-recovery path."""

    fingerprint = "fake-fit"

    def __init__(self):
        self.idata = FakeIdata(CANNED_POSTERIOR)


class CannedPredictionFit:
    """A fitted model that predicts a spread of p_left, for the holdout path.

    The predictions vary across stimuli (a constant would make every Pearson
    correlation undefined) and are identical for every model, so a test that
    recovers the ground truth reads r == 1.0 exactly.
    """

    model = None

    def predict_p_left(self, stim_data):
        return np.linspace(0.1, 0.9, stim_data["n"])
