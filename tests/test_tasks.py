"""Each project's task: its response schema, prediction and stimulus pools.

The pipeline used to hard-code the coin-flip task; a `TaskSpec` per project
carries what differs, found by project id or, where only a response file is at
hand, by its columns (a response file carries exactly its task's raw columns).
"""

from __future__ import annotations

import pytest

from src.tasks import task_for_project, task_present_in


def test_subjective_randomness_keeps_its_schema():
    task = task_for_project("subjective_randomness")
    assert task.raw_columns == ("sequence_a", "sequence_b", "participant_id", "trial_index", "chose_left")
    assert task.stimulus_columns == ("sequence_a", "sequence_b")
    assert task.response_column == "chose_left"
    assert task.prediction_var == "p_left"
    # Left/right is counterbalanced: both presentations are one stimulus.
    assert task.stimulus_key({"sequence_a": "HT", "sequence_b": "HH"}) == task.stimulus_key(
        {"sequence_a": "HH", "sequence_b": "HT"}
    )


def test_number_game_schema():
    task = task_for_project("number_game")
    assert task.raw_columns == ("number_set", "target", "participant_id", "trial_index", "rating")
    assert task.response_column == "rating" and task.prediction_var == "p_yes"
    assert task.stimulus_key({"number_set": "16,8", "target": "4"}) != task.stimulus_key(
        {"number_set": "8,16", "target": "4"}
    )


def test_unknown_project_raises():
    with pytest.raises(KeyError, match="no task"):
        task_for_project("think_aloud_game24")


def test_number_game_pools():
    task = task_for_project("number_game")
    design = task.design_pool()
    assert len(design) == 255 * 100
    assert len({task.stimulus_key(r) for r in design}) == len(design)
    novelty = task.novelty_pool()
    assert len(novelty) == 512 and novelty == task.novelty_pool()  # fixed seed
    assert set(novelty[0]) == {"number_set", "target"}


def test_a_file_with_extra_columns_still_names_its_task():
    stimulus = {"number_set": "16", "target": 4, "eig": 0.1, "selection_rank": 1}
    assert task_present_in(stimulus, "stimulus_columns").name == "number_game"
    with pytest.raises(ValueError, match="stimulus columns"):
        task_present_in({"eig": 0.1}, "stimulus_columns")


def test_number_game_agent_texts_carry_no_coin_flips_and_name_no_seed():
    """What number-game agents read: no coin-flip leftovers, and neither seed's
    authors (one of the two is the held-out ground truth in a recovery cell)."""
    from src.runtime.config import PROJECT_ASSETS_DIR

    task = task_for_project("number_game")
    texts = {
        "theory prompt": task.theory_prompt.read_text(),
        "critique prompt": task.critique_prompt.read_text(),
        "task description": (PROJECT_ASSETS_DIR / "number_game" / "task_description.md").read_text(),
        "lenses": "\n".join(task.candidate_hints),
    }
    for name, text in texts.items():
        lowered = text.lower()
        for word in ("sequence_a", "chose_left", "p_left", "h/t", "heads", "tenenbaum", "bigelow", "piantadosi"):
            assert word not in lowered, f"{name} mentions {word!r}"
    assert "p_yes" in texts["theory prompt"] and "rating" in texts["critique prompt"]
