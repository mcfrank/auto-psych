from src.pipelines.outer_loop.deployment.firestore import (
    metadata_documents,
    responses_to_csv,
    validate_submit_payload,
)
from src.pipelines.outer_loop.deployment.manifest import DeploymentManifest


def _manifest():
    return DeploymentManifest(
        project_id="subjective_randomness",
        experiment_id="subjective_randomness_experiment1",
        run_id=1,
        deployment_id="deploy_1",
        collection_session_id="session_1",
        study_id="study_subjective_randomness",
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
    )


def test_metadata_documents_use_session_centered_paths():
    docs = metadata_documents(_manifest())
    assert "studies/study_subjective_randomness" in docs
    assert "deployments/deploy_1" in docs
    assert "collection_sessions/session_1" in docs
    assert docs["collection_sessions/session_1"]["collection_owner"] == "linas"


def test_validate_submit_payload_accepts_session_or_legacy_shape():
    assert validate_submit_payload({"collection_session_id": "s", "trials": []}) == (True, "")
    assert validate_submit_payload({"project_id": "p", "run_id": 1, "trials": []}) == (True, "")
    ok, msg = validate_submit_payload({"project_id": "p", "trials": []})
    assert not ok
    assert "collection_session_id" in msg


def test_responses_to_csv_quotes_and_flattens_trials():
    csv_text = responses_to_csv(
        [
            (
                "participant,1",
                {
                    "trials": [
                        {"sequence_a": "H,H", "sequence_b": "HT", "chose_left": True},
                        {"sequence_a": "TT", "sequence_b": "HH", "chose_left": False},
                    ]
                },
            )
        ]
    )
    assert "participant_id,participant_id_str" in csv_text
    assert '"participant,1"' in csv_text
    assert '"H,H"' in csv_text
    assert ",1,0," in csv_text
