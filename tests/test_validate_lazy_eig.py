"""The lazy-vs-exact EIG validation script reads a finished cell without
writing into it, and scores every set on shared fresh scenarios."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.paths import SCRIPTS_DIR, load_script_module

validate = load_script_module(SCRIPTS_DIR / "subjective_randomness" / "validate_lazy_eig.py")


def _cell(tmp_path):
    cell = tmp_path / "cell_1"
    (cell / "experiment2" / "cognitive_models").mkdir(parents=True)
    (cell / "experiment2" / "cognitive_models" / "m.py").write_text("x = 1\n")
    (cell / "experiment2" / "design" / "_fit_cache").mkdir(parents=True)
    (cell / "experiment2" / "design" / "stimuli.json").write_text("[]")
    (cell / "experiment1" / "model_loop").mkdir(parents=True)
    (cell / "experiment1" / "model_loop" / "responses.csv").write_text("a\n")
    (cell / "experiment1" / "model_registry.yaml").write_text("theories: {}\n")
    return cell


def test_inputs_are_copied_out_of_the_cell(tmp_path):
    cell = _cell(tmp_path)
    staged = validate.stage_inputs(cell, tmp_path / "work", 2)
    assert (staged / "experiment2" / "cognitive_models" / "m.py").read_text() == "x = 1\n"
    assert (staged / "experiment1" / "model_loop" / "responses.csv").exists()
    assert (staged / "experiment2" / "design" / "stimuli.json").exists()


def test_a_work_dir_inside_the_cell_is_refused(tmp_path):
    cell = _cell(tmp_path)
    with pytest.raises(ValueError, match="inside the cell"):
        validate.stage_inputs(cell, cell / "validation", 2)


def test_the_sweeps_pairs_map_back_to_pool_indices():
    pool = [{"sequence_a": "HH", "sequence_b": "HT"}, {"sequence_a": "HT", "sequence_b": "TH"}]
    assert validate.pool_index_of(pool, [{"sequence_a": "HT", "sequence_b": "TH"}]) == [1]
    with pytest.raises(ValueError):
        validate.pool_index_of(pool, [{"sequence_a": "TT", "sequence_b": "HT"}])


def test_every_set_is_scored_against_the_reference_on_shared_scenarios():
    rng = np.random.default_rng(0)
    draws = {m: rng.uniform(0.05, 0.95, size=(10, 40)) for m in ("a", "b")}
    args = validate.Args(cell=None, work_dir=None, fresh_scenarios=400, n_responses=3)
    sets = {
        "ref": {"indices": [0, 1, 2, 3], "n_response_picks": 2},
        "same": {"indices": [0, 1, 2, 3], "n_response_picks": 2},
        "other": {"indices": [5, 6, 7, 8], "n_response_picks": 0},
    }
    validate.score_sets(draws, None, sets, "ref", args)
    assert sets["same"]["fresh_diff_vs_reference"] == 0.0
    assert sets["same"]["fresh_bits"] == sets["ref"]["fresh_bits"]
    assert sets["other"]["overlap_with_reference"] == 0
    assert sets["other"]["fresh_bits_n_response_picks"] is None
    assert sets["ref"]["fresh_bits_n_response_picks"] is not None
    json.dumps(sets)
