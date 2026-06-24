"""Unit tests for posterior-mean extraction in fit_mega_models.py.

The mega-analysis records, alongside each model's fit metrics, the posterior
mean (and sd) of every fitted parameter. These tests cover the pure extraction
helper — building a small InferenceData by hand, no MCMC — including that
vector-valued parameters (e.g. a Dirichlet ``weights``) are unpacked
element-wise and that non-parameter posterior variables (the per-trial
Deterministic ``p_left``) are excluded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "analysis" / "behavioral" / "fit_mega_models.py"


def _load_cli():
    """Load the standalone analysis script as a module (its helpers are the units)."""
    spec = importlib.util.spec_from_file_location("fit_mega_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _toy_idata():
    """A hand-built posterior with a scalar param, a length-3 vector param, and a
    per-trial Deterministic stand-in (`p_left`) that is NOT a fitted parameter."""
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    chains, draws = 2, 100
    posterior = {
        "theta_alt": rng.normal(0.5, 0.01, size=(chains, draws)),
        "weights": rng.normal([0.2, 0.3, 0.5], 0.01, size=(chains, draws, 3)),
        "p_left": rng.uniform(0.0, 1.0, size=(chains, draws, 4)),
    }
    return az.from_dict(posterior=posterior)


def test_scalar_and_vector_params_extracted():
    """Scalars stay scalar; a vector param is unpacked into name[0], name[1], ..."""
    rows = cli.posterior_param_means(_toy_idata(), ["theta_alt", "weights"])
    by_param = {r["param"]: r for r in rows}

    assert set(by_param) == {"theta_alt", "weights[0]", "weights[1]", "weights[2]"}
    assert by_param["theta_alt"]["posterior_mean"] == pytest.approx(0.5, abs=0.02)
    assert by_param["weights[0]"]["posterior_mean"] == pytest.approx(0.2, abs=0.02)
    assert by_param["weights[2]"]["posterior_mean"] == pytest.approx(0.5, abs=0.02)
    assert by_param["theta_alt"]["posterior_sd"] > 0


def test_only_requested_params_returned():
    """The per-trial Deterministic ``p_left`` is not a fitted parameter and must be
    excluded — only the names passed in (the model's free RVs) appear."""
    rows = cli.posterior_param_means(_toy_idata(), ["theta_alt", "weights"])
    assert all(not r["param"].startswith("p_left") for r in rows)
