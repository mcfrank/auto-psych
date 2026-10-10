"""A live study excludes everyone who took part in an earlier study of the
pipeline (October 2026: two people did experiments 1 and 2 of the same run,
and nothing kept run 1's 118 people out of runs 2 and 3).

Prolific's ``previous_studies_blocklist`` filter takes study IDs and is
evaluated when the study is published, so the IDs are read from the account
at creation time (``GET /studies/``), never written into a config: the
pipeline's own studies are the ones whose ``internal_name`` it set.
"""

import pytest

import src.runtime.prolific as prolific_client
from src.pipelines.outer_loop.deployment import prolific as deploy_prolific
from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest

PROJECT = "subjective_randomness"


@pytest.fixture
def prolific(tmp_path, monkeypatch):
    """Prolific stand-ins: the account's studies, its filters, and a create
    call that records the payload it was sent."""
    monkeypatch.setattr(
        prolific_client, "project_assets_dir", lambda pid: tmp_path / pid
    )
    (tmp_path / PROJECT).mkdir()
    settings = tmp_path / PROJECT / "prolific_config.yaml"
    settings.write_text("{}\n", encoding="utf-8")
    account = {
        "studies": [
            {"id": "run1_e1", "status": "AWAITING REVIEW",
             "internal_name": "auto-psych deploy_subjective_randomness-e1-run1"},
            {"id": "run1_e2", "status": "COMPLETED",
             "internal_name": "auto-psych deploy_subjective_randomness-e2-run1"},
            {"id": "run2_e1", "status": "ACTIVE",
             "internal_name": "auto-psych deploy_subjective_randomness-e1-run2"},
            # A draft never had participants.
            {"id": "draft", "status": "UNPUBLISHED",
             "internal_name": "auto-psych deploy_subjective_randomness-e1-run3"},
            # Someone else's study in the same account.
            {"id": "other", "status": "COMPLETED", "internal_name": "lab survey"},
        ],
        "list_error": None,
        "created": [],
    }
    monkeypatch.setattr(
        prolific_client,
        "list_studies",
        lambda: (None, account["list_error"])
        if account["list_error"]
        else (list(account["studies"]), None),
    )
    monkeypatch.setattr(
        deploy_prolific, "verify_live_eligibility", lambda: None
    )

    def create_study(payload):
        account["created"].append(payload)
        return "new_study", None

    monkeypatch.setattr(prolific_client, "create_study", create_study)
    account["settings"] = settings
    return account


def _manifest() -> DeploymentManifest:
    return DeploymentManifest(
        project_id=PROJECT,
        experiment_id="subjective_randomness_experiment1",
        run_id=3,
        deployment_id="deploy_subjective_randomness-e1-run3",
        collection_session_id="session_1",
        study_id="study_subjective_randomness",
        deploy_target="firebase",
        prolific_mode="live",
        agent_backend="claude",
        collection_owner="tester",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        experiment_url="https://example.org/exp",
        results_api_url="https://example.org/exp",
    )


def _blocklist(payload):
    by_id = {f["filter_id"]: f for f in payload["filters"]}
    return by_id.get("previous_studies_blocklist")


def test_a_live_study_excludes_the_participants_of_every_earlier_pipeline_study(
    prolific,
):
    plan = deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")

    (payload,) = prolific["created"]
    assert _blocklist(payload)["selected_values"] == ["run1_e1", "run1_e2", "run2_e1"]
    # The data-quality filters are still there.
    assert {f["filter_id"] for f in payload["filters"]} >= {
        "current-country-of-residence", "fluent-languages", "approval_rate",
    }
    # What was sent is what the manifest records.
    assert plan.payload == payload
    assert plan.study_id == "new_study"


def test_the_listing_matches_the_name_the_pipeline_gives_its_studies(prolific):
    plan = deploy_prolific.build_prolific_plan(
        project_id=PROJECT, manifest=_manifest(), n_participants=40, mode="live"
    )
    assert plan.payload["internal_name"].startswith(
        deploy_prolific.STUDY_INTERNAL_NAME_PREFIX
    )


def test_a_study_listing_failure_creates_no_study(prolific):
    prolific["list_error"] = "GET /studies/ 503: down"
    with pytest.raises(RuntimeError, match="503"):
        deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")
    assert prolific["created"] == []


def test_the_first_study_has_no_blocklist(prolific):
    prolific["studies"] = [prolific["studies"][-1]]  # only someone else's study
    deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")
    (payload,) = prolific["created"]
    assert _blocklist(payload) is None


def test_a_test_draft_excludes_no_one(prolific):
    deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "test")
    (payload,) = prolific["created"]
    assert _blocklist(payload) is None


def test_a_series_can_recruit_earlier_participants_on_purpose(prolific):
    prolific["settings"].write_text(
        "exclude_earlier_participants: false\n", encoding="utf-8"
    )
    deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")
    (payload,) = prolific["created"]
    assert _blocklist(payload) is None


def test_exclude_earlier_participants_must_be_a_boolean(prolific):
    prolific["settings"].write_text(
        "exclude_earlier_participants: 'no'\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exclude_earlier_participants"):
        deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")
    assert prolific["created"] == []


def test_live_eligibility_requires_the_blocklist_filter():
    filters = [
        {"filter_id": "current-country-of-residence", "choices": {"1": "United States"}},
        {"filter_id": "fluent-languages", "choices": {"19": "English"}},
    ]
    with pytest.raises(ValueError, match="previous_studies_blocklist"):
        deploy_prolific.verify_eligibility_choice_ids(filters)
    deploy_prolific.verify_eligibility_choice_ids(
        filters + [{"filter_id": "previous_studies_blocklist", "data_type": "StudyID"}]
    )


def test_a_project_can_exclude_only_its_own_earlier_participants(prolific, tmp_path):
    # The RSA campaign (PI 2026-10-10): subjective-randomness participants may take part.
    (tmp_path / "rsa_reference").mkdir()
    (tmp_path / "rsa_reference" / "prolific_config.yaml").write_text(
        "exclude_earlier_participants_from: project\n", encoding="utf-8")
    prolific["studies"] += [
        {"id": "rsa_c0_e1", "status": "COMPLETED", "internal_name": "auto-psych deploy_rsa_reference-e1-c0-x"},
        {"id": "rsa_c1_e1", "status": "ACTIVE", "internal_name": "auto-psych deploy_rsa_reference-e1-c1-x"},
        {"id": "rsa_draft", "status": "UNPUBLISHED", "internal_name": "auto-psych deploy_rsa_reference-e2-c0-x"},
    ]
    deploy_prolific.create_draft_study("rsa_reference", _manifest(), 200, "live")
    (payload,) = prolific["created"]
    assert _blocklist(payload)["selected_values"] == ["rsa_c0_e1", "rsa_c1_e1"]


def test_the_exclusion_scope_is_all_or_project(prolific):
    prolific["settings"].write_text("exclude_earlier_participants_from: sr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'all' or 'project'"):
        deploy_prolific.create_draft_study(PROJECT, _manifest(), 40, "live")
