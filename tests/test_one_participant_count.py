"""The number recruited, the design's N and the poll target are one setting.

Prolific places came from the project's ``prolific_config.yaml``
(``total_available_places``, with a hidden default of 1 when absent) while
``--n-participants`` set the design's N and the poll target. The launchers
rendered the two from one config; a direct ``run.py`` call did not, and the
committed render (40 places) recruited and paid 40 people for a run that
stopped waiting after ``--n-participants``. Now ``--n-participants`` is the
only source: the payload's places are that number, a config that names a
different one raises before any stage runs, and a missing config (no render)
raises instead of creating a study from defaults.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

import src.runtime.prolific as prolific_client
from src.pipelines.outer_loop import run as outer_run
from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest
from src.pipelines.outer_loop.deployment.prolific import build_prolific_plan
from tests.paths import REPO_ROOT, SCRIPTS_DIR, load_script_module

PROJECT = "subjective_randomness"
STUDY_SETTINGS = {
    "reward_per_hour": 1200,
    "estimated_completion_time": 7,
    "name": "Which sequence is more random?",
    "completion_code": "CODE",
}


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """A project assets dir whose prolific_config.yaml the test writes."""
    monkeypatch.setattr(
        prolific_client, "project_assets_dir", lambda pid: tmp_path / pid
    )
    (tmp_path / PROJECT).mkdir()

    def write(settings):
        (tmp_path / PROJECT / "prolific_config.yaml").write_text(
            yaml.safe_dump(settings), encoding="utf-8"
        )

    return write


def _manifest() -> DeploymentManifest:
    return DeploymentManifest(
        project_id=PROJECT,
        experiment_id=f"{PROJECT}_experiment1",
        run_id=1,
        deployment_id="deploy_1",
        collection_session_id="session_1",
        study_id=f"study_{PROJECT}",
        deploy_target="dry-run",
        prolific_mode="live",
        agent_backend="claude",
        collection_owner="tester",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        experiment_url="https://example.org/exp",
        results_api_url="https://example.org/exp",
    )


def _plan(n_participants):
    return build_prolific_plan(
        project_id=PROJECT,
        manifest=_manifest(),
        n_participants=n_participants,
        mode="live",
    )


def test_the_study_recruits_n_participants(assets):
    assets(STUDY_SETTINGS)
    assert _plan(7).payload["total_available_places"] == 7


def test_a_config_that_names_another_count_raises(assets):
    assets({**STUDY_SETTINGS, "total_available_places": 40})
    with pytest.raises(ValueError, match=r"40.*--n-participants 5"):
        _plan(5)


def test_a_config_that_agrees_is_accepted(assets):
    assets({**STUDY_SETTINGS, "total_available_places": 5})
    assert _plan(5).payload["total_available_places"] == 5


def test_no_rendered_config_raises_instead_of_using_defaults(assets):
    with pytest.raises(FileNotFoundError, match="--render-only"):
        _plan(5)


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


def _live_args(n_participants):
    return outer_run.Args(
        project=PROJECT,
        experiment=1,
        mode="live",
        deploy_target="firebase",
        prolific_mode="live",
        confirm_live_recruitment=True,
        n_participants=n_participants,
        coding_agent="claude",
        firebase_project="auto-psych-test",
    )


def test_run_py_refuses_a_disagreeing_config_before_any_stage(
    assets, stages_run, tmp_path, monkeypatch
):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CODING_AGENT", "claude")
    assets({**STUDY_SETTINGS, "total_available_places": 40})

    with pytest.raises(ValueError, match="40"):
        outer_run.main(_live_args(5))

    assert stages_run == []
    assert not (tmp_path / "output").exists()


def test_run_py_refuses_a_missing_config_before_any_stage(
    assets, stages_run, tmp_path, monkeypatch
):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CODING_AGENT", "claude")

    with pytest.raises(FileNotFoundError):
        outer_run.main(_live_args(5))

    assert stages_run == []


def test_the_launchers_render_no_participant_count(tmp_path, monkeypatch):
    pilot_config = load_script_module(
        SCRIPTS_DIR / "outer_loop_live" / "_pilot_config.py"
    )
    monkeypatch.setattr(pilot_config, "project_assets_dir", lambda pid: tmp_path / pid)
    (tmp_path / PROJECT).mkdir()
    config = tmp_path / "pilot.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": PROJECT,
                "run_label": "t",
                "prolific": {"participants": 10, **STUDY_SETTINGS},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["_pilot_config.py", str(config), "--render-only"])

    pilot_config.main()

    rendered = yaml.safe_load((tmp_path / PROJECT / "prolific_config.yaml").read_text())
    assert "total_available_places" not in rendered
    assert rendered["reward_per_hour"] == 1200


def test_no_rendered_config_is_committed():
    committed = subprocess.run(
        [
            "git",
            "ls-files",
            f"src/pipelines/outer_loop/projects/{PROJECT}/prolific_config.yaml",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert committed == ""
