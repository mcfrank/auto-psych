"""A run can hide ``participant_id`` from everything agents and models see.

With ``hide_participant_id: true`` in a holdout config, the agent-facing
responses carry the task's raw columns without ``participant_id`` (as in
auto-rsa, whose models see only the display a trial showed): every model is
population-level, with no participant random effect. ``participant_id`` is
the one optional raw column; a response file carries it on every row or on
none, and the task description agents read lists only the columns present.
"""

from __future__ import annotations

import pytest

from src.pipelines.outer_loop.columns import raw_response_rows
from src.pipelines.outer_loop.model_loop_runner import write_task_description
from src.tasks import task_for_columns, task_for_project, task_present_in

NG = task_for_project("number_game")
ROW = {"number_set": "16,8", "target": "4", "participant_id": "0", "trial_index": "0", "rating": "1"}
HIDDEN = {k: v for k, v in ROW.items() if k != "participant_id"}


def test_a_file_without_participant_id_still_names_its_task():
    assert task_for_columns(list(HIDDEN)).name == "number_game"
    assert task_present_in(list(HIDDEN)).name == "number_game"
    with pytest.raises(ValueError, match="no task"):
        task_for_columns(["number_set", "target", "participant_id", "rating"])  # trial_index is required


def test_raw_rows_keep_participant_id_only_when_the_rows_carry_it():
    assert list(raw_response_rows([{**ROW, "model": "gt"}], NG)[0]) == list(ROW)
    assert list(raw_response_rows([{**HIDDEN, "model": "gt"}], NG)[0]) == list(HIDDEN)
    with pytest.raises(ValueError, match="participant_id"):
        raw_response_rows([HIDDEN, ROW], NG)


def test_the_task_description_lists_only_the_columns_present(tmp_path):
    with_ids = write_task_description("number_game", tmp_path / "a", columns=list(ROW)).read_text()
    without = write_task_description("number_game", tmp_path / "b", columns=list(HIDDEN)).read_text()
    assert "`participant_id`" in with_ids
    assert "`participant_id`" not in without and "`trial_index`" in without
    assert len(with_ids.splitlines()) == len(without.splitlines()) + 1


_PER_PERSON = '''import numpy as np
import pymc as pm


def compute_features(number_set, target):
    return {"even": float(int(target) % 2 == 0)}


with pm.Model() as model:
    even = pm.Data("even", np.zeros(2))
    participant_id = pm.Data("participant_id", np.zeros(2, dtype="int64"))
    rating = pm.Data("rating", np.zeros(2, dtype="int64"))
    b = pm.Normal("b", 0, 1)
    u = pm.Normal("u", 0, 1, shape=50)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(b * even + u[participant_id]))
    pm.Bernoulli("obs", p=p_yes, observed=rating)
'''


def test_a_participant_effect_is_rejected_when_the_column_is_hidden(tmp_path):
    """Rejected with a reason the repair slot reads, not a crash of the cell."""
    from src.pipelines.inner_loop.model_zoo import _cheap_gate_rejection

    (tmp_path / "candidate.py").write_text(_PER_PERSON)
    responses = tmp_path / "responses.csv"
    responses.write_text("number_set,target,trial_index,rating\n16,4,0,1\n16,5,1,0\n3,9,2,1\n")
    models = tmp_path / "models"
    models.mkdir()
    reason = _cheap_gate_rejection(tmp_path / "candidate.py", "a bias per person", models, "pp", responses)
    assert "missing columns ['participant_id']" in reason
