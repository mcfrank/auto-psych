from src.pipelines.outer_loop.deployment.firestore import (
    responses_to_csv,
    validate_submit_payload,
)


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
