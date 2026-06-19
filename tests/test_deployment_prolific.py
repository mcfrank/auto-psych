from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest
from src.pipelines.outer_loop.deployment.prolific import (
    build_prolific_plan,
    completion_redirect_url,
    compute_reward_cents,
    external_study_url,
)


def test_compute_reward_from_hourly_wage_for_five_minutes():
    # $12.00/hr == 1200 cents/hr; a 5-minute study pays 1200 * 5/60 = 100 cents.
    assert compute_reward_cents({"reward_per_hour": 1200, "estimated_completion_time": 5}) == 100


def test_compute_reward_scales_with_duration():
    assert compute_reward_cents({"reward_per_hour": 1200, "estimated_completion_time": 10}) == 200


def test_compute_reward_uses_explicit_reward_when_no_hourly_rate():
    assert compute_reward_cents({"reward": 75}) == 75


def test_compute_reward_falls_back_to_default():
    assert compute_reward_cents({}) == 50


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
