"""Live deploys work from the launchers' run copies, in a coherent order.

1. **Provenance without ``.git``.** ``run_pilot.sh`` and
   ``submit_parallel.sh`` run each job from an rsync copy of the checkout
   made with ``--exclude '.git'``, and the deploy's provenance step raised
   outside a git checkout, so no launcher-driven study could deploy. The
   launchers now record the checkout's commit and whether its tree was dirty
   (untracked files included) into the copy (``code_provenance.json``), and
   the deploy reads that record when there is no ``.git``; with neither it
   still raises.
2. **Deploy first, then the Prolific draft.** The draft used to be created
   and recorded before the Firebase deploy, so a failed deploy left a
   recorded live study that the "never publish a second study" guard then
   refused to relaunch past. Now the page and functions are deployed and
   verified live first; only then is the draft created, recorded at once, and
   (live mode only) published. A failed deploy leaves no study on record.
3. **``prolific_mode: none`` through the launchers** deploys and stops before
   collection (there is no study to collect from), as test mode does.

Prolific and Firebase are mocked; nothing is contacted.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from src.pipelines.outer_loop import run as outer_run
from src.pipelines.outer_loop.deployment import local
from src.pipelines.outer_loop.deployment import prolific as deploy_prolific
from src.pipelines.outer_loop.deployment.local import run_deployment
from src.pipelines.outer_loop.deployment.manifest import (
    CODE_PROVENANCE_FILENAME,
    code_provenance,
    manifest_path,
    record_code_provenance,
    refuse_second_live_study,
)
from src.pipelines.outer_loop.deployment.prolific import ProlificStudyPlan
from tests.paths import REPO_ROOT
from tests.test_live_study_relaunch_guard import PROJECT, _live_args, live_run_dir, stages_run  # noqa: F401

LAUNCHERS = ("run_pilot.sh", "submit_parallel.sh")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def _copy(tmp_path, checkout):
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "code.py").write_text((checkout / "code.py").read_text(), encoding="utf-8")
    return copy


# ── 1. Provenance in a copy without .git ────────────────────────────────


def test_a_run_copy_takes_its_provenance_from_the_launchers_record(tmp_path):
    checkout = _checkout(tmp_path)
    copy = _copy(tmp_path, checkout)

    record = record_code_provenance(checkout, copy)

    assert record == copy / CODE_PROVENANCE_FILENAME
    provenance = code_provenance(copy)
    assert provenance["git_commit"] == _git(checkout, "rev-parse", "HEAD")
    assert provenance["git_dirty"] is False
    assert str(checkout) in provenance["source"]


def test_an_untracked_file_in_the_checkout_makes_the_record_dirty(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "new_idea.py").write_text("y = 2\n", encoding="utf-8")
    copy = _copy(tmp_path, checkout)

    record_code_provenance(checkout, copy)

    assert code_provenance(copy)["git_dirty"] is True


def test_a_copy_with_neither_git_nor_a_record_refuses_to_deploy(tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    with pytest.raises(RuntimeError, match=CODE_PROVENANCE_FILENAME):
        code_provenance(copy)


def test_a_malformed_record_is_refused(tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / CODE_PROVENANCE_FILENAME).write_text('{"git_commit": "abc"}', encoding="utf-8")
    with pytest.raises(ValueError, match="git_dirty"):
        code_provenance(copy)


def test_a_git_checkout_is_read_from_git_not_from_a_record(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / CODE_PROVENANCE_FILENAME).write_text(
        json.dumps({"git_commit": "0" * 40, "git_dirty": False, "source": "stale"}),
        encoding="utf-8",
    )
    provenance = code_provenance(checkout)
    assert provenance["git_commit"] == _git(checkout, "rev-parse", "HEAD")
    assert provenance["source"] == "git checkout"


def test_recording_refuses_a_checkout_that_is_not_a_git_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="cannot record deployment provenance"):
        record_code_provenance(tmp_path, tmp_path)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_each_launcher_records_provenance_right_after_copying(launcher):
    text = (REPO_ROOT / "scripts" / "outer_loop_live" / launcher).read_text(encoding="utf-8")
    copy_at = text.index("rsync -a --delete")
    record_at = text.index("src.pipelines.outer_loop.deployment.record_provenance")
    submit_at = text.index("sbatch --parsable")
    assert copy_at < record_at < submit_at


def test_the_recorder_cli_writes_the_record(tmp_path):
    checkout = _checkout(tmp_path)
    copy = _copy(tmp_path, checkout)
    from src.pipelines.outer_loop.deployment.record_provenance import Args, main

    main(Args(checkout=checkout, copy=copy))

    assert code_provenance(copy)["git_commit"] == _git(checkout, "rev-parse", "HEAD")


# ── 2. The deploy order: page first, then the draft ─────────────────────


@pytest.fixture
def mocked_services(tmp_path, monkeypatch):
    """Firebase and Prolific stand-ins that record the order of calls."""
    calls = []
    monkeypatch.setattr(local, "results_token", lambda: "token")
    monkeypatch.setattr(local, "write_functions_env", lambda repo_root: None)
    monkeypatch.setattr(
        local, "run_firebase_deploy", lambda *a, **k: calls.append("firebase deploy")
    )
    monkeypatch.setattr(
        local, "verify_functions_live", lambda m: calls.append("verify functions")
    )

    def create_draft(project_id, manifest, n_participants, mode):
        calls.append("create draft")
        return ProlificStudyPlan(
            payload={"name": "study"}, completion_code="CODE",
            redirect_url="https://app.prolific.com/submissions/complete?cc=CODE",
            study_id="draft-1",
        )

    def publish(plan):
        calls.append("publish")
        plan.published = True
        return plan

    monkeypatch.setattr(local, "create_draft_study", create_draft)
    monkeypatch.setattr(local, "publish_study", publish)
    monkeypatch.setattr(
        local, "verify_live_eligibility", lambda: calls.append("check eligibility")
    )
    # The study settings a launcher renders into the copy.
    monkeypatch.setattr(
        "src.runtime.prolific.project_assets_dir", lambda pid: tmp_path / "assets" / pid
    )
    (tmp_path / "assets" / PROJECT).mkdir(parents=True)
    (tmp_path / "assets" / PROJECT / "prolific_config.yaml").write_text(
        "reward: 100\n", encoding="utf-8"
    )
    # Staging writes public/ under the repo root: use a copy with a record.
    repo = tmp_path / "run_copy"
    repo.mkdir()
    (repo / CODE_PROVENANCE_FILENAME).write_text(
        json.dumps({"git_commit": "a" * 40, "git_dirty": False, "source": "test"}),
        encoding="utf-8",
    )
    return calls, repo


def _experiment(tmp_path):
    exp_dir = tmp_path / "experiment1"
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "experiment" / "index.html").write_text("<html></html>", encoding="utf-8")
    return exp_dir


def _firebase_deploy(exp_dir, repo, prolific_mode):
    return run_deployment(
        exp_dir=exp_dir, project_id=PROJECT, run_id=1, deploy_target="firebase",
        prolific_mode=prolific_mode, agent_backend="claude", collection_owner="tester",
        firebase_project="auto-psych-test", firebase_region="us-central1",
        n_participants=2, repo_root=repo, run_label="run1",
    )


def test_a_live_deploy_puts_the_page_up_before_creating_and_publishing_the_study(
    tmp_path, mocked_services
):
    calls, repo = mocked_services
    exp_dir = _experiment(tmp_path)

    _firebase_deploy(exp_dir, repo, "live")

    assert calls == [
        "check eligibility", "firebase deploy", "verify functions", "create draft", "publish",
    ]
    manifest = json.loads(manifest_path(exp_dir).read_text())
    assert manifest["prolific_study_id"] == "draft-1"
    assert manifest["metadata"]["prolific_published"] is True
    assert manifest["git_commit"] == "a" * 40
    assert manifest["metadata"]["code_provenance"] == "test"
    # Collection reads the study id from the experiment's config.
    config = json.loads((exp_dir / "experiment" / "config.json").read_text())
    assert config["prolific_study_id"] == "draft-1"
    # The deployed page redirects participants with the completion code.
    assert config["prolific_redirect_url"].endswith("cc=CODE")


def test_a_test_deploy_creates_an_unpublished_draft_after_the_page(tmp_path, mocked_services):
    calls, repo = mocked_services
    exp_dir = _experiment(tmp_path)
    _firebase_deploy(exp_dir, repo, "test")
    assert calls == ["firebase deploy", "verify functions", "create draft"]


def test_a_failed_firebase_deploy_leaves_no_study_and_a_plain_relaunch_works(
    tmp_path, monkeypatch, mocked_services
):
    calls, repo = mocked_services
    exp_dir = _experiment(tmp_path)

    def broken(*a, **k):
        raise RuntimeError("Firebase deploy failed: token expired")

    monkeypatch.setattr(local, "run_firebase_deploy", broken)
    with pytest.raises(RuntimeError, match="token expired"):
        _firebase_deploy(exp_dir, repo, "live")

    assert "create draft" not in calls
    assert json.loads(manifest_path(exp_dir).read_text())["prolific_study_id"] is None
    refuse_second_live_study(exp_dir, refused="deploy again")  # does not raise

    monkeypatch.setattr(
        local, "run_firebase_deploy", lambda *a, **k: calls.append("firebase deploy")
    )
    _firebase_deploy(exp_dir, repo, "live")
    assert calls[-3:] == ["verify functions", "create draft", "publish"]


def test_live_eligibility_is_checked_before_anything_is_deployed(
    tmp_path, monkeypatch, mocked_services
):
    calls, repo = mocked_services

    def drifted():
        raise ValueError("Prolific eligibility choice IDs changed")

    monkeypatch.setattr(local, "verify_live_eligibility", drifted)
    with pytest.raises(ValueError, match="choice IDs changed"):
        _firebase_deploy(_experiment(tmp_path), repo, "live")
    assert calls == []


def test_verify_live_eligibility_reads_prolifics_filters(monkeypatch):
    import src.runtime.prolific as prolific_client

    monkeypatch.setattr(prolific_client, "get_filters", lambda: (None, "HTTP 503"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        deploy_prolific.verify_live_eligibility()


# ── 3. prolific_mode none through the launchers ─────────────────────────


def test_prolific_mode_none_deploys_and_stops_before_collection(
    live_run_dir, stages_run, monkeypatch  # noqa: F811
):
    def implement(**kwargs):
        stages_run.append("3_implement")
        return True, ""

    monkeypatch.setattr(outer_run, "spawn_cc_agent", implement)
    outer_run.main(
        _live_args(prolific_mode="none", confirm_live_recruitment=False, experiment=None,
                   experiments="2")
    )
    assert stages_run == ["2_design", "3_implement", "deploy"]


def test_a_resume_with_prolific_mode_none_still_collects(live_run_dir, stages_run):  # noqa: F811
    outer_run.main(
        _live_args(prolific_mode="none", confirm_live_recruitment=False, agent="4_collect")
    )
    assert stages_run == ["4_collect"]



def test_the_pilot_preset_ships_in_test_mode():
    """Its comment always said test was the safe default; it shipped as live."""
    import yaml

    preset = yaml.safe_load(
        (REPO_ROOT / "scripts" / "outer_loop_live" / "pilot.yaml").read_text(encoding="utf-8")
    )
    assert preset["prolific_mode"] == "test"
    assert not preset.get("confirm_live_recruitment")
