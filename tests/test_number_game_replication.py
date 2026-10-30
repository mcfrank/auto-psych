"""The number-game seeds replicate Bigelow & Piantadosi (2016) on its dataset.

"Inferring priors in compositional cognitive models" (CogSci 2016) fits the
prior of its Mixture and Independent models to these participants' ratings,
after dropping the 31 least typical participants (258,750 responses). The
seeds are those two models as LOTlib's grammars define them (the paper's
implementation, `LOTlib/Legacy/GrammarInferenceVectorized/Examples/NumberGame`).
Its Figure 3 gives the fitted mixture parameter lambda_0, ~0.355 for the
Independent model and ~0.24 for the Mixture (read off the plot), and ranks the
Independent model's rule priors: even numbers, multiples of 11, multiples of 5,
odd numbers. Both posteriors are narrow, so the MAP stands in for them here.

Figure 2's R^2 (0.982 and 0.989 at the MAP) is not reproduced: these models
give 0.96-0.98, and no lambda_0 brings the Mixture Model to 0.982, so the gap
lies in how that figure was computed, not in the model.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.number_game import data as ng_data
from src.number_game.metrics import binned_mean_r2
from src.runtime.config import REPO_ROOT

DATA_CSV = ng_data.DATA_CSV
SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"

needs_data = pytest.mark.skipif(not DATA_CSV.exists(), reason="number-game dataset not present")


@needs_data
def test_trimming_leaves_the_papers_response_count():
    responses = ng_data.trim_least_typical(ng_data.load_responses(DATA_CSV), n_trimmed=31)
    assert len(responses) == 258_750
    assert responses["id"].nunique() == 575


def test_binned_r2_is_computed_on_bin_means():
    cells = pd.DataFrame({"yes": [0, 1, 5, 9], "n": [10, 10, 10, 10]})
    p_model = np.array([0.01, 0.04, 0.52, 0.93])
    # Bins [0, .05) -> human mean .05, [.5, .55) -> .5, [.9, .95) -> .9.
    expected = np.corrcoef([0.025, 0.525, 0.925], [0.05, 0.5, 0.9])[0, 1] ** 2
    assert binned_mean_r2(p_model, cells) == pytest.approx(expected)


def _seed_module(seed):
    spec = importlib.util.spec_from_file_location(seed, SEEDS_DIR / f"{seed}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper_map(seed):
    """The seed's MAP on the paper's trimmed responses."""
    responses = ng_data.trim_least_typical(ng_data.load_responses(DATA_CSV), n_trimmed=31)
    rows = [
        {"number_set": ",".join(map(str, s)), "target": t, "rating": r}
        for s, t, r in zip(responses["set"], responses["target"], responses["rating"])
    ]
    model = load_pymc_model(seed, SEEDS_DIR)
    with model:
        pm.set_data(make_stim_data(model, rows))
        return pm.find_MAP(progressbar=False)


@pytest.fixture(scope="module")
def independent_map():
    return _paper_map("bigelow_piantadosi_independent")


@needs_data
def test_independent_lambda_0_is_figure_3s(independent_map):
    assert float(independent_map["lambda_0"]) == pytest.approx(0.355, abs=0.02)


@needs_data
def test_independent_ranks_the_rule_priors_as_figure_3(independent_map):
    names = _seed_module("bigelow_piantadosi_independent").RULE_NAMES
    top = [names[i] for i in np.argsort(-independent_map["rule_weights"])[:4]]
    assert top == ["even", "multiples of 11", "multiples of 5", "odd"]


@needs_data
def test_mixture_lambda_0_is_figure_3s():
    assert float(_paper_map("bigelow_piantadosi_mixture")["lambda_0"]) == pytest.approx(0.24, abs=0.02)
