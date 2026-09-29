"""A run that already has a live Prolific study never creates another one.

``run_live.sbatch`` always passes ``--resume``, so relaunching a run (after a
crash, or by reusing its label) re-ran every stage: a new design, a new
experiment deployed over the one participants were taking, and a second
Prolific study created and published — recruiting and paying a second group.
The experiment's deployment manifest records the study; a run that finds one
refuses the stages that would redo it, and says how to recover.
"""

from __future__ import annotations

import json

import pytest

import src.runtime.prolific as prolific_client
from src.pipelines.outer_loop import orchestrator
from src.pipelines.outer_loop import run as outer_run
from src.pipelines.outer_loop.deployment.local import run_deployment
from src.pipelines.outer_loop.deployment.manifest import (
    DeploymentManifest,
    LiveStudyAlreadyRecorded,
    manifest_dir,
    manifest_path,
    write_manifest,
)
from tests.paths import REPO_ROOT

PROJECT = "subjective_randomness"
LIVE_STUDY = "study-live-1"


@pytest.fixture
def no_prolific(monkeypatch):
    """Every call that creates, publishes or looks up a study fails the test."""

    def contacted(*args, **kwargs):
        raise AssertionError("the pipeline contacted Prolific")

    for name in ("create_study", "publish_study", "get_filters"):
        monkeypatch.setattr(prolific_client, name, contacted)


def _record_study(exp_dir, *, prolific_mode="live", study_id=LIVE_STUDY, published=True):
    manifest = DeploymentManifest(
        project_id=PROJECT,
        experiment_id=f"{PROJECT}_experiment1",
        run_id=1,
        deployment_id="deploy_first",
        collection_session_id="session_first",
        study_id=f"study_{PROJECT}",
        deploy_target="firebase",
        prolific_mode=prolific_mode,
        agent_backend="claude",
        collection_owner="tester",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        experiment_url="https://auto-psych-test.web.app/e1-run1/",
        results_api_url="https://auto-psych-test.web.app",
        hosting_path="e1-run1",
        prolific_study_id=study_id,
        total_available_places=2,
        metadata={"prolific_published": True} if published else {},
    )
    write_manifest(exp_dir, manifest)


def _live_args(**overrides) -> outer_run.Args:
    settings = dict(
        project=PROJECT,
        experiment=1,
        mode="live",
        deploy_target="firebase",
        prolific_mode="live",
        confirm_live_recruitment=True,
        firebase_project="auto-psych-test",
        run_label="run1",
        n_participants=2,
        coding_agent="claude",
        resume=True,
    )
    settings.update(overrides)
    return outer_run.Args(**settings)


@pytest.fixture
def live_run_dir(tmp_path, monkeypatch):
    # The study settings a launcher renders (run.py checks them up front).
    monkeypatch.setattr(prolific_client, "project_assets_dir", lambda pid: tmp_path / pid)
    (tmp_path / PROJECT).mkdir()
    (tmp_path / PROJECT / "prolific_config.yaml").write_text("reward: 100\n", encoding="utf-8")
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CODING_AGENT", "claude")
    exp_dir = orchestrator.experiment_dir(PROJECT, 1)
    (exp_dir / "experiment").mkdir(parents=True)
    return exp_dir


@pytest.fixture
def stages_run(monkeypatch):
    """Stand-ins for every stage, recording which ones ran."""
    ran = []

    def recorder(name):
        return lambda *a, **k: ran.append(name)

    monkeypatch.setattr(outer_run, "run_design_programmatic", recorder("2_design"))
    monkeypatch.setattr(outer_run, "spawn_cc_agent", recorder("3_implement"))
    monkeypatch.setattr(outer_run, "run_deployment_programmatic", recorder("deploy"))
    monkeypatch.setattr(outer_run, "run_collect_programmatic", recorder("4_collect"))
    monkeypatch.setattr(outer_run, "run_inner_model_loop_programmatic", recorder("5_model_loop"))
    monkeypatch.setattr(outer_run, "begin_model_loop_stage", lambda *a, **k: None)
    monkeypatch.setattr(outer_run, "finish_model_loop_stage", recorder("registry"))
    return ran


def test_relaunching_a_run_with_a_live_study_refuses_before_any_stage(
    live_run_dir, stages_run, no_prolific
):
    _record_study(live_run_dir)

    with pytest.raises(LiveStudyAlreadyRecorded) as excinfo:
        outer_run.main(_live_args())

    assert stages_run == []
    message = str(excinfo.value)
    assert LIVE_STUDY in message
    assert "RESUME_AGENTS" in message
    assert "--publish-another-prolific-study" in message


@pytest.mark.parametrize("stage", ["2_design", "3_implement"])
def test_rerunning_design_or_implement_alone_is_refused_too(
    live_run_dir, stage, stages_run, no_prolific
):
    _record_study(live_run_dir)
    with pytest.raises(LiveStudyAlreadyRecorded):
        outer_run.main(_live_args(agent=stage))
    assert stages_run == []


def test_deploy_only_is_refused(live_run_dir, stages_run, no_prolific):
    _record_study(live_run_dir)
    with pytest.raises(LiveStudyAlreadyRecorded):
        outer_run.main(_live_args(deploy_only=True))
    assert stages_run == []


def test_a_study_whose_publish_was_not_confirmed_still_blocks(
    live_run_dir, stages_run, no_prolific
):
    # The publish call may have gone through on Prolific's side even though the
    # run never recorded it (a timeout on the response, a crash right after).
    _record_study(live_run_dir, published=False)
    with pytest.raises(LiveStudyAlreadyRecorded, match="may have been published"):
        outer_run.main(_live_args())
    assert stages_run == []


@pytest.mark.parametrize("stage", ["4_collect", "5_model_loop"])
def test_resume_agents_route_finishes_from_the_existing_study(
    live_run_dir, stage, stages_run, no_prolific
):
    _record_study(live_run_dir)
    outer_run.main(_live_args(agent=stage))
    assert stages_run[0] == stage


def test_a_test_mode_draft_does_not_block_a_relaunch(live_run_dir, stages_run, no_prolific):
    _record_study(live_run_dir, prolific_mode="test", study_id="draft-1", published=False)
    outer_run.main(_live_args(agent="2_design", prolific_mode="test"))
    assert stages_run == ["2_design"]


def _deploy(exp_dir, **kwargs):
    return run_deployment(
        exp_dir=exp_dir,
        project_id=PROJECT,
        run_id=1,
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="tester",
        firebase_project=None,
        firebase_region="us-central1",
        n_participants=2,
        repo_root=REPO_ROOT,
        run_label="run1",
        **kwargs,
    )


def _experiment_to_deploy(exp_dir):
    (exp_dir / "experiment").mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment" / "index.html").write_text("<html></html>", encoding="utf-8")


def test_the_deployment_itself_refuses_to_deploy_over_a_live_study(tmp_path, no_prolific):
    exp_dir = tmp_path / "experiment1"
    _experiment_to_deploy(exp_dir)
    _record_study(exp_dir)

    with pytest.raises(LiveStudyAlreadyRecorded):
        _deploy(exp_dir)

    assert json.loads(manifest_path(exp_dir).read_text())["prolific_study_id"] == LIVE_STUDY


def test_the_override_deploys_and_keeps_the_earlier_study_on_record(tmp_path, no_prolific):
    exp_dir = tmp_path / "experiment1"
    _experiment_to_deploy(exp_dir)
    _record_study(exp_dir)

    _deploy(exp_dir, publish_another_prolific_study=True)

    superseded = sorted(manifest_dir(exp_dir).glob("deployment_manifest.superseded-*.json"))
    assert len(superseded) == 1
    assert json.loads(superseded[0].read_text())["prolific_study_id"] == LIVE_STUDY
    assert json.loads(manifest_path(exp_dir).read_text())["deployment_id"] != "deploy_first"
