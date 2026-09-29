"""The researchers' raw collected file never lands where agents can read it.

Collection keeps every collected column -- Prolific IDs, and in simulations
the generating model's name -- in ``<output>/<project>/raw_collected/``. On a
default ``run.py`` run the output tree is ``REPO_ROOT/data/outer_loop``, and
``run.py``'s agents run with the repository as their (read-only) working
tree, so every agent could read it. ``run.py`` now refuses to start a run
that collects with its output inside the repository, before any stage, and
collection refuses to write the file there.
"""

from __future__ import annotations

import io

import pytest

from src.pipelines.outer_loop import collect
from src.pipelines.outer_loop import orchestrator
from src.pipelines.outer_loop import run as outer_run

PROJECT = "subjective_randomness"
RESULTS_CSV = (
    "participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model\n"
    "0,5f8a1b2c3d4e5f6a7b8c9d0e,0,HHT,THT,1,0,\n"
    "0,5f8a1b2c3d4e5f6a7b8c9d0e,1,HTHT,HHHH,0,1,\n"
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A stand-in repository: the agents' working tree."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(orchestrator, "REPO_ROOT", root)
    return root


def test_collection_refuses_to_write_the_raw_file_inside_the_agents_tree(
    repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(repo / "data" / "outer_loop"))
    exp_dir = orchestrator.experiment_dir(PROJECT, 1)
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "experiment" / "config.json").write_text(
        '{"prolific_study_id": "study1", "results_api_url": "http://results.invalid",'
        ' "total_available_places": 1}',
        encoding="utf-8",
    )
    monkeypatch.setattr(collect, "_poll_prolific_until_target", lambda *a, **k: 1)
    monkeypatch.setattr(
        collect.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(RESULTS_CSV.encode()),
    )

    with pytest.raises(RuntimeError, match="AUTO_PSYCH_OUTPUT_DIR"):
        orchestrator.run_collect_programmatic(
            exp_dir,
            mode="live",
            n_participants=1,
            project_id=PROJECT,
            prolific_mode="live",
        )

    assert not orchestrator.raw_collected_responses_path(exp_dir).parent.exists()


def test_collection_outside_the_agents_tree_is_unchanged(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    orchestrator.require_outside_agent_trees(
        tmp_path / "output" / PROJECT / "raw_collected", "raw data"
    )


@pytest.fixture
def stages_run(monkeypatch):
    ran = []
    for name in (
        "run_design_programmatic",
        "spawn_cc_agent",
        "run_deployment_programmatic",
        "run_collect_programmatic",
        "run_inner_model_loop_programmatic",
    ):
        monkeypatch.setattr(outer_run, name, lambda *a, _n=name, **k: ran.append(_n))
    return ran


def _args(**overrides):
    return outer_run.Args(
        project=PROJECT,
        experiment=1,
        n_participants=2,
        coding_agent="claude",
        **overrides,
    )


def test_run_py_refuses_an_output_tree_inside_the_repository_before_any_stage(
    repo, stages_run, monkeypatch
):
    monkeypatch.delenv("AUTO_PSYCH_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(
        orchestrator, "outer_data_dir", lambda: repo / "data" / "outer_loop"
    )
    monkeypatch.setattr(
        outer_run, "outer_data_dir", lambda: repo / "data" / "outer_loop"
    )
    monkeypatch.setenv("CODING_AGENT", "claude")

    with pytest.raises(RuntimeError, match="inside .*repo"):
        outer_run.main(_args())

    assert stages_run == []
    assert not (repo / "data").exists()


def test_run_py_without_collection_is_not_refused(
    repo, stages_run, monkeypatch, tmp_path
):
    """A model-stage-only rerun writes no raw file."""
    monkeypatch.setattr(
        outer_run, "outer_data_dir", lambda: repo / "data" / "outer_loop"
    )
    monkeypatch.setattr(outer_run, "_run_experiment", lambda **k: None)
    monkeypatch.setenv("CODING_AGENT", "claude")
    monkeypatch.setenv("CLAUDE_AUTH", "subscription")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token")
    outer_run.main(_args(agent="5_model_loop"))
