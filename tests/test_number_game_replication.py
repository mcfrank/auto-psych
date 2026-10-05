"""The number-game seed replicates Bigelow & Piantadosi (2016) on its dataset.

"Inferring priors in compositional cognitive models" (CogSci 2016) fits the
rules + intervals model's grammar prior to these 606 participants' ratings.
Its Figure 2 reports R^2 over the means of human p(yes) binned by model
probability: 0.976-0.989 for the mixture and independent-rule grammars. Its
Figure 3 ranks the independent model's rule priors: even numbers, multiples of
11, multiples of 5, odd numbers. The seed is Eric Bigelow's numpyro version of
that model (ported exactly, `test_number_game_seed_port.py`), fitted to the
untrimmed cell counts as his code does. Its posterior is narrow (p_mix sd
0.003 under NUTS), so the MAP stands in for it here; the full posterior gives
R^2 0.978 too (`scripts/number_game/check_pymc_port.py compare`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.number_game import data as ng_data
from src.number_game.metrics import binned_mean_r2
from src.runtime.config import REPO_ROOT

DATA_CSV = REPO_ROOT / "data" / "number-game" / "human experiment" / "numbergame_data.csv"
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


@pytest.fixture(scope="module")
def seed_map():
    responses = ng_data.load_responses(DATA_CSV)
    rows = [
        {"number_set": ",".join(map(str, s)), "target": t, "rating": r}
        for s, t, r in zip(responses["set"], responses["target"], responses["rating"])
    ]
    model = load_pymc_model("bayesian_rules_intervals", SEEDS_DIR)
    with model:
        pm.set_data(make_stim_data(model, rows))
        point = pm.find_MAP(progressbar=False)
    cells = ng_data.cell_counts(responses)
    by_cell = pd.DataFrame({
        "set": responses["set"], "target": responses["target"], "p_yes": point["p_yes"],
    }).groupby(["set", "target"], sort=True)["p_yes"].first()
    return point, cells, by_cell.to_numpy()


@needs_data
def test_seed_fit_is_in_the_papers_r2_range(seed_map):
    _, cells, p_yes = seed_map
    assert binned_mean_r2(p_yes, cells) == pytest.approx(0.978, abs=0.003)


@needs_data
def test_seed_ranks_the_rule_priors_as_figure_3(seed_map):
    point, _, _ = seed_map
    even, multiples_of_11, multiples_of_5, odd = 38, 17, 11, 39  # the seed's RULES order
    assert list(np.argsort(-point["rule_weights"])[:4]) == [even, multiples_of_11, multiples_of_5, odd]
