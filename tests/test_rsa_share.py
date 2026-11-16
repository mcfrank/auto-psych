"""The de-identified sharing of live trial data (src/rsa/share.py)."""

import pytest

from src.rsa.share import (UnknownField, check_no_ids, deidentify_trials, person_hash, pseudonym_of,
                           submission_ids)

PID = "5f3a9c0e1b2d4f6a8c7e9b01"  # Prolific-shaped, made up
SESSION = "65aa00bb11cc22dd33ee44ff"


def record(**extra):
    return {"trial_type": "html-button-response", "task": "experience", "rt": 1234, "participant_id": PID,
            "stimulus": "<p>How much did you enjoy it?</p>", "plugin_version": "2.0.0", "rating": 4,
            "comment": "my name is ...", "objects": [[0, 1], [1, 0]], **extra}


def test_hash_is_keyed_stable_and_distinct():
    a, b = person_hash(b"k1", PID), person_hash(b"k1", PID)
    assert a == b and a.startswith("h") and PID not in a
    assert person_hash(b"k2", PID) != a
    assert person_hash(b"k1", SESSION) != a
    with pytest.raises(ValueError):
        person_hash(b"", PID)


def test_free_text_ids_and_html_dropped_structure_kept():
    (row,) = deidentify_trials([record()], "habc")
    assert row["person"] == "habc" and row["rating"] == 4 and row["rt"] == 1234
    assert row["objects"] == "[[0, 1], [1, 0]]"
    for gone in ("comment", "participant_id", "stimulus", "plugin_version"):
        assert gone not in row


def test_unknown_field_raises():
    with pytest.raises(UnknownField, match="free_response"):
        deidentify_trials([record(free_response="typed")], "habc")


def test_check_no_ids_catches_real_and_prolific_shaped_ids():
    ids = submission_ids({"prolific_pid": PID, "prolific_session_id": SESSION}, [record()])
    assert ids == {PID, SESSION}
    check_no_ids([{"person": "habc", "design_sha256": "ab" * 32}], ids)  # a sha256 is not an id
    with pytest.raises(ValueError):
        check_no_ids([{"person": "habc", "x": f"seen {PID}"}], ids)
    with pytest.raises(ValueError):
        check_no_ids([{"person": "habc", "x": "0123456789abcdef01234567"}], ids)


def test_pseudonym_of_both_map_formats():
    assert pseudonym_of({PID: "live7"}, PID) == "live7"
    assert pseudonym_of({PID: {"id": "p42", "first_study": "x"}}, PID) == "p42"
    assert pseudonym_of({}, PID) is None
