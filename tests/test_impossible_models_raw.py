"""The impossible ground-truth models generate data from raw rows.

Under the raw-only pipeline data rows carry only ``sequence_a`` and
``sequence_b``, and every model computes its own features. The four impossible
models still declared the old featurizer's columns (``h_a``/``h_b``,
``max_run_norm_a/b``, ``imbalance_a/b``) with no ``compute_features`` hook, so
every cell of the 2026-09-28 impossible sweep stopped at data generation with
``MissingStimulusColumns: Rows missing columns ['h_a', 'h_b']``. Each now has
the hook, computing exactly what the old featurizer
(``src/subjective_randomness/features.py``) computed.
"""

from __future__ import annotations

import ast
from itertools import product

import numpy as np
import pytest

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model, model_source_file, pm_data_inputs
from src.subjective_randomness import features
from src.subjective_randomness.holdout_data import (
    generate_responses,
    p_left_fixed_params,
)
from tests.paths import REPO_ROOT

IMPOSSIBLE_DIR = REPO_ROOT / "src" / "subjective_randomness" / "impossible_models"
MODELS = [
    "more_heads_more_random",
    "fewer_heads_more_random",
    "longer_runs_more_random",
    "more_imbalance_more_random",
]
SEQUENCES = ["".join(s) for n in range(1, 9) for s in product("HT", repeat=n)]
# Every sequence against a fixed partner of each length: every sequence of
# lengths 1-8 on both sides.
PAIRS = [(a, b) for a in SEQUENCES for b in ("H", "TH", "HHT", "HTHTTHTH")] + [
    ("HTTH", s) for s in SEQUENCES
]
STIMULI = [
    {"sequence_a": a, "sequence_b": b}
    for a, b in product(["HHTT", "HTHT", "HHHH"], ["THTT", "TTTT", "HTTH"])
]


def _function_source(path, name):
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{path} defines no {name}")


@pytest.mark.parametrize("name", MODELS)
def test_compute_features_matches_the_old_featurizer(name):
    model = load_pymc_model(name, IMPOSSIBLE_DIR)
    columns = sorted(set(pm_data_inputs(model)) - {"chose_left"})
    hook = getattr(model, "_auto_psych_compute_features")
    for a, b in PAIRS:
        computed = hook(a, b)
        old = features.featurize_stimulus(a, b)
        assert sorted(computed) == columns
        for column in columns:
            assert computed[column] == pytest.approx(old[column], abs=0, rel=1e-12), (
                a,
                b,
                column,
            )
            assert type(computed[column]) is type(old[column]), (a, b, column)


@pytest.mark.parametrize("name", MODELS)
def test_the_vendored_clean_sequence_is_the_featurizers(name):
    assert _function_source(
        IMPOSSIBLE_DIR / f"{name}.py", "clean_sequence"
    ) == _function_source(
        REPO_ROOT / "src" / "subjective_randomness" / "features.py", "clean_sequence"
    )


@pytest.mark.parametrize("name", MODELS)
def test_each_impossible_model_generates_data_from_raw_rows(name):
    params = {"beta": 4.0, "side_bias": 0.0}
    p_left = p_left_fixed_params(name, IMPOSSIBLE_DIR, STIMULI, params, seed=3)
    assert p_left.shape == (len(STIMULI),)
    assert np.all((p_left > 0) & (p_left < 1))
    assert np.ptp(p_left) > 0  # the rule separates these pairs

    rows = generate_responses(
        name, IMPOSSIBLE_DIR, STIMULI, params, 3, seed=3, participant_id_offset=0
    )
    assert len(rows) == 3 * len(STIMULI)
    assert set(rows[0]) == {
        "sequence_a",
        "sequence_b",
        "participant_id",
        "trial_index",
        "chose_left",
        "generating_model",
    }


def test_the_score_directions_are_the_rules():
    """p_left > 0.5 exactly when the left sequence scores higher under the rule
    (side_bias 0)."""
    rows = [
        {"sequence_a": "HHHT", "sequence_b": "HTTT"},  # more heads, same run
        {"sequence_a": "HHHH", "sequence_b": "HTHT"},
    ]  # longer run, more imbalance
    params = {"beta": 4.0, "side_bias": 0.0}
    p = {
        name: p_left_fixed_params(name, IMPOSSIBLE_DIR, rows, params) for name in MODELS
    }
    assert p["more_heads_more_random"][0] > 0.5 > p["fewer_heads_more_random"][0]
    assert p["longer_runs_more_random"][1] > 0.5
    assert p["more_imbalance_more_random"][1] > 0.5
    assert p["more_imbalance_more_random"][0] == pytest.approx(0.5)


def test_binding_a_raw_row_needs_no_precomputed_column():
    for name in MODELS:
        model = load_pymc_model(name, IMPOSSIBLE_DIR)
        assert model_source_file(model).name == f"{name}.py"
        make_stim_data(model, [{"sequence_a": "H", "sequence_b": "T", "chose_left": 0}])
