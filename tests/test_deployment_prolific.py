from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest
from src.pipelines.outer_loop.deployment.prolific import (
    build_prolific_plan,
    completion_redirect_url,
    external_study_url,
)


def test_external_study_url_adds_prolific_params():
    url = external_study_url("https://example.org/exp")
    assert "participant_id={{%PROLIFIC_PID%}}" in url
    assert "STUDY_ID={{%STUDY_ID%}}" in url


def test_completion_redirect_url_escapes_code():
    assert completion_redirect_url("A B") == "https://app.prolific.com/submissions/complete?cc=A%20B"


def test_build_prolific_payload_without_network():
    manifest = DeploymentManifest(
        project_id="subjective_randomness",
        experiment_id="subjective_randomness_experiment1",
        run_id=1,
        deployment_id="deploy_1",
        collection_session_id="session_1",
        study_id="study_subjective_randomness",
        deploy_target="dry-run",
        prolific_mode="test",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        experiment_url="https://example.org/exp",
        results_api_url="https://example.org/exp",
    )

    plan = build_prolific_plan(
        project_id="missing_project_uses_defaults",
        manifest=manifest,
        n_participants=7,
        mode="test",
        test_participant_id="tester",
    )

    assert plan.payload["external_study_url"].startswith("https://example.org/exp?")
    assert plan.payload["total_available_places"] == 1
    assert plan.payload["filters"][0]["selected_values"] == ["tester"]
    assert plan.completion_code == "AUTO_PSYCH_COMPLETE"
