"""The holdout configs' production fits: 1000 draws per chain, and enough
posterior draws for everything that consumes them.

Draws went from 2000 to 1000 per chain on 2026-09-27 to bring a cell under
12 hours. The critique draws its posterior-predictive replicates from the
posterior (capped at chains x draws), and the evaluation thins the posterior
to ``eval_pool.predict_max_draws``; neither may be starved by the cut.
"""

from __future__ import annotations

import pytest
import yaml

from src.pipelines.inner_loop.critique_round import CRITIQUE_PPC_REPLICATES
from tests.paths import SCRIPTS_DIR

CONFIGS = [
    "holdout_recovery_faithful.yaml",
    "holdout_recovery_faithful_motif_only.yaml",
    "holdout_recovery.yaml",
    "holdout_recovery_deepseek.yaml",
    "impossible_holdout_recovery.yaml",
]


@pytest.mark.parametrize("name", CONFIGS)
def test_production_fits_draw_1000_per_chain_and_enough_in_all(name):
    config = yaml.safe_load((SCRIPTS_DIR / "subjective_randomness" / "configs" / name).read_text())
    fit = config["fit"]
    assert (fit["draws"], fit["tune"], fit["chains"]) == (1000, 1000, 4)
    posterior_draws = fit["draws"] * fit["chains"]
    assert posterior_draws >= CRITIQUE_PPC_REPLICATES
    assert posterior_draws >= config["eval_pool"]["predict_max_draws"]
