"""Inner-loop agents are told the task.

Before this, candidate and critique agents saw the data columns but were never
told what a trial is or what ``chose_left`` means. The project's
``task_description.md`` (checked to name no model or mechanism) is copied next
to the inner loop's responses and inlined into both agents' context.
"""

from __future__ import annotations

import pytest

from src.pipelines.inner_loop.candidate_agent import _write_candidate_context
from src.pipelines.inner_loop.critique_round import _write_critique_context
from src.pipelines.outer_loop.model_loop_runner import write_task_description

TASK = "# The task\n\nParticipants pick the sequence that looks more random.\n"


def _loop(tmp_path, with_task=True):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: seed_a\n    rationale: mechanism seed_a\n", encoding="utf-8"
    )
    responses = tmp_path / "responses.csv"
    responses.write_text("sequence_a,sequence_b,chose_left\nHT,HH,1\n", encoding="utf-8")
    if with_task:
        (tmp_path / "task_description.md").write_text(TASK, encoding="utf-8")
    return models_dir, responses


def test_the_candidate_context_carries_the_task(tmp_path):
    models_dir, responses = _loop(tmp_path)
    docs = _write_candidate_context(
        tmp_path / "iter_0" / "candidate_0", responses, models_dir,
        iteration=0, candidate_idx=0, candidate_count=3, current_posterior=None,
    )
    assert "Participants pick the sequence that looks more random." in docs["context"]


def test_the_critique_context_carries_the_task(tmp_path):
    models_dir, responses = _loop(tmp_path)
    text = _write_critique_context(
        tmp_path / "critique", "seed_a", models_dir, responses, tmp_path / "cache",
        n_proposals=8, significance_alpha=0.05, n_replicates=10,
    )
    assert "Participants pick the sequence that looks more random." in text


def test_a_missing_task_description_fails_loudly(tmp_path):
    models_dir, responses = _loop(tmp_path, with_task=False)
    with pytest.raises(FileNotFoundError, match="task_description.md"):
        _write_candidate_context(
            tmp_path / "iter_0" / "candidate_0", responses, models_dir,
            iteration=0, candidate_idx=0, candidate_count=3, current_posterior=None,
        )


def test_the_outer_loop_copies_the_projects_task_description(tmp_path):
    written = write_task_description("subjective_randomness", tmp_path)
    assert written == tmp_path / "task_description.md"
    assert "chose_left" in written.read_text(encoding="utf-8")


def test_a_project_without_one_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="task_description.md"):
        write_task_description("no_such_project", tmp_path)


def test_the_shipped_description_names_no_ground_truth_or_mechanism():
    from src.pipelines.outer_loop.orchestrator import outer_project_dir

    text = (outer_project_dir("subjective_randomness") / "task_description.md").read_text()
    for word in ("motif", "stack", "falk", "konold", "occurrence", "representativ",
                 "griffiths", "symmetr", "mirror", "alternat", "run length", "streak"):
        assert word not in text.lower(), word
