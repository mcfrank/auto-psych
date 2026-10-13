"""The number-game PyMC seeds are faithful ports of Eric Bigelow's numpyro models:
`bigelow_piantadosi_independent` of `prior_type='basic'` and
`bigelow_piantadosi_interval` of `prior_type='interval'` (with the reference's
interval-midpoint index corrected, `check_pymc_port.corrected_midpoint_grid`).

`scripts/number_game/check_pymc_port.py` evaluates the numpyro reference
(`scripts/number_game/numpyro_reference/`, run under `uv run --with numpyro
--with jax`, 64-bit) at parameter points drawn from its prior and writes the
joint log density and the predictive per cell to a golden file. The port must
reproduce both. The reference observes a Binomial per (set, target) cell, the
port a Bernoulli per response, so the log densities differ by the constant
sum of log C(n, yes), which the golden file records.

The port samples the normalised rule weights w ~ Dirichlet(2, ..., 2) rather
than the reference's raw r_i ~ Gamma(2, 1): only w = r / s enters the
likelihood, and the raw total s = sum r (~ Gamma(2K, 1), independent of w) is
identified by the prior alone and mixed slowly (bulk ESS 85 over 4,000 PyMC
draws). With the change of variables r -> (s, w), whose Jacobian is s^(K-1),

    log p_ref(r) = log p_port(w) + log Gamma(s; 2K, 1) - (K - 1) log s.
"""

from __future__ import annotations

import json

import numpy as np
import pymc as pm
import pytest
from scipy.stats import gamma

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.runtime.config import REPO_ROOT
from tests.paths import FIXTURES_DIR

SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
# seed -> the reference's prior type (its golden file).
PORTS = {"bigelow_piantadosi_independent": "basic", "bigelow_piantadosi_interval": "interval"}


def _rows(cells):
    """One raw response row per yes and per no of every cell, as the CSV gives them."""
    rows = []
    for number_set, target, yes, no in cells:
        for rating in [1] * yes + [0] * no:
            rows.append({
                "number_set": number_set,
                "target": str(target),
                "participant_id": "0",
                "trial_index": str(len(rows)),
                "rating": str(rating),
            })
    return rows


@pytest.fixture(scope="module", params=list(PORTS))
def port(request):
    """(golden, the seed bound to the golden cells)."""
    golden = json.loads((FIXTURES_DIR / "number_game" / f"numpyro_{PORTS[request.param]}_golden.json").read_text())
    model = load_pymc_model(request.param, SEEDS_DIR)
    with model:
        pm.set_data(make_stim_data(model, _rows(golden["cells"])))
    return golden, model


def _value_point(model, point):
    """Transformed-space values for the reference's constrained parameters."""
    raw = np.asarray(point["p_rules"])
    constrained = {"p_mix": np.float64(point["p_mix"]), "rule_weights": raw / raw.sum()}
    if "p_mean_hyper" in point:
        constrained["midpoint_shape"] = np.asarray(point["p_mean_hyper"])
    values = {}
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms[rv]
        x = constrained[rv.name]
        values[value_var.name] = transform.forward(x, *rv.owner.inputs).eval() if transform else x
    return values


def test_log_density_matches_numpyro(port):
    golden, bound_model = port
    logp = bound_model.compile_logp(jacobian=False)
    for point in golden["points"]:
        raw = np.asarray(point["p_rules"])
        total, k = raw.sum(), len(raw)
        scale_density = gamma.logpdf(total, 2 * k) - (k - 1) * np.log(total)
        expected = point["log_density"] - golden["log_binomial_coefficients"] - scale_density
        assert logp(_value_point(bound_model, point)) == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_predictions_match_numpyro_per_response(port):
    golden, bound_model = port
    [p_yes_graph] = bound_model.replace_rvs_by_values([bound_model["p_yes"]])
    p_yes = bound_model.compile_fn(p_yes_graph, inputs=bound_model.value_vars, on_unused_input="ignore")
    n_per_cell = [yes + no for _, _, yes, no in golden["cells"]]
    for point in golden["points"]:
        expected = np.repeat(point["post_predictive"], n_per_cell)
        np.testing.assert_allclose(p_yes(_value_point(bound_model, point)), expected, rtol=1e-9, atol=1e-12)
