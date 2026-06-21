import pytest

from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest
from src.pipelines.outer_loop.deployment.prolific import (
    build_eligibility_filters,
    build_prolific_plan,
    completion_redirect_url,
    compute_reward_cents,
    external_study_url,
    verify_eligibility_choice_ids,
)


def _live_filters_snapshot() -> list[dict]:
    """A minimal stand-in for Prolific's GET /filters/ payload covering the
    filters whose choice IDs we hardcode."""
    return [
        {
            "filter_id": "current-country-of-residence",
            "choices": {"0": "United Kingdom", "1": "United States"},
        },
        {
            "filter_id": "fluent-languages",
            "choices": {"0": "Rather not say", "19": "English"},
        },
        {"filter_id": "approval_rate", "type": "range"},
    ]


def _manifest() -> DeploymentManifest:
    return DeploymentManifest(
        project_id="subjective_randomness",
        experiment_id="subjective_randomness_experiment1",
        run_id=1,
        deployment_id="deploy_1",
        collection_session_id="session_1",
        study_id="study_subjective_randomness",
        deploy_target="dry-run",
        prolific_mode="live",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        experiment_url="https://example.org/exp",
        results_api_url="https://example.org/exp",
    )


def _filters_by_id(payload) -> dict:
    return {f["filter_id"]: f for f in payload.get("filters", [])}


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


def test_eligibility_filters_default_to_us_english_and_98_percent():
    filters = {f["filter_id"]: f for f in build_eligibility_filters({})}
    assert filters["current-country-of-residence"]["selected_values"] == ["1"]
    assert filters["fluent-languages"]["selected_values"] == ["19"]
    assert filters["approval_rate"]["selected_range"] == {"lower": 98, "upper": 100}


def test_eligibility_filters_respect_configured_min_approval_rate():
    filters = {f["filter_id"]: f for f in build_eligibility_filters({"min_approval_rate": 90})}
    assert filters["approval_rate"]["selected_range"] == {"lower": 90, "upper": 100}


def test_eligibility_filters_reject_out_of_range_approval_rate():
    with pytest.raises(ValueError):
        build_eligibility_filters({"min_approval_rate": 150})


def test_verify_eligibility_choice_ids_passes_on_expected_mapping():
    # Should not raise when the live snapshot matches our hardcoded IDs.
    verify_eligibility_choice_ids(_live_filters_snapshot())


def test_verify_eligibility_choice_ids_raises_on_drift():
    drifted = _live_filters_snapshot()
    drifted[0]["choices"]["1"] = "Uruguay"  # choice "1" no longer means United States
    with pytest.raises(ValueError, match="current-country-of-residence"):
        verify_eligibility_choice_ids(drifted)


def test_verify_eligibility_choice_ids_raises_on_missing_filter():
    without_language = [f for f in _live_filters_snapshot() if f["filter_id"] != "fluent-languages"]
    with pytest.raises(ValueError, match="fluent-languages"):
        verify_eligibility_choice_ids(without_language)


def test_live_study_applies_data_quality_eligibility_filters():
    # Integration: a real-recruitment (live) study restricts to US-based,
    # English-fluent participants with an approval rate of at least 98%.
    plan = build_prolific_plan(
        project_id="missing_project_uses_defaults",
        manifest=_manifest(),
        n_participants=10,
        mode="live",
    )
    filters = _filters_by_id(plan.payload)

    assert filters["current-country-of-residence"]["selected_values"] == ["1"]
    assert filters["fluent-languages"]["selected_values"] == ["19"]
    assert filters["approval_rate"]["selected_range"] == {"lower": 98, "upper": 100}
