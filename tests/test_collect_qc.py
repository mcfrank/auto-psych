"""Tests for the collected-data quality guard.

The simulated-participant collector must fail loudly when it returns data with
no response variation (e.g. every trial chose the same side) — that data can't
support model comparison and almost always means the steering, not the
participant, decided every trial.
"""

from __future__ import annotations

import pytest

from src.pipelines.outer_loop.collect import check_response_variation


def _rows(chose_left_values):
    return [
        {"participant_id": i, "sequence_a": "HT", "sequence_b": "TH", "chose_left": v}
        for i, v in enumerate(chose_left_values)
    ]


def test_varied_responses_pass():
    ok, msg = check_response_variation(_rows([1, 0, 1, 0]))
    assert ok, msg


def test_all_identical_responses_fail():
    ok, msg = check_response_variation(_rows([1, 1, 1, 1]))
    assert not ok
    assert "identical" in msg


def test_all_identical_zeros_also_fail():
    ok, _ = check_response_variation(_rows([0, 0, 0]))
    assert not ok


def test_string_values_from_csv_are_handled():
    # Firebase /results CSV yields chose_left as strings.
    ok, _ = check_response_variation(_rows(["1", "1", "1"]))
    assert not ok
    ok2, _ = check_response_variation(_rows(["1", "0", "1"]))
    assert ok2


def test_no_rows_fail():
    ok, msg = check_response_variation([])
    assert not ok
    assert "no response" in msg.lower()


def test_single_row_is_not_flagged_as_degenerate():
    # One row has no variation to speak of; don't false-alarm on tiny inputs.
    ok, _ = check_response_variation(_rows([1]))
    assert ok


# ── Integration: the guard is wired into the active programmatic collector ──


def _make_exp_dir(tmp_path):
    import json

    exp_dir = tmp_path / "experiment1"
    (exp_dir / "design").mkdir(parents=True)
    (exp_dir / "design" / "stimuli.json").write_text(
        json.dumps([{"sequence_a": "HT", "sequence_b": "TH"}]), encoding="utf-8"
    )
    (exp_dir / "experiment").mkdir()
    (exp_dir / "experiment" / "config.json").write_text(
        json.dumps(
            {
                "results_api_url": "https://example.invalid",
                "experiment_url": "https://example.invalid",
                "collection_session_id": "sess_test",
                "project_id": "subjective_randomness",
                "run_id": 1,
            }
        ),
        encoding="utf-8",
    )
    return exp_dir


def test_programmatic_collect_raises_on_degenerate_firebase_data(tmp_path, monkeypatch):
    from src.pipelines.outer_loop import collect, orchestrator

    monkeypatch.setattr(
        collect, "_collect_from_firebase", lambda *a, **k: _rows([1, 1, 1, 1])
    )
    with pytest.raises(RuntimeError, match="quality check"):
        orchestrator.run_collect_programmatic(
            _make_exp_dir(tmp_path),
            mode="simulated_participants",
            n_participants=2,
            project_id="subjective_randomness",
        )


def test_poll_prolific_times_out_instead_of_hanging(tmp_path, monkeypatch):
    # A study that never reaches its target must not loop forever.
    from src.pipelines.outer_loop import collect
    import src.runtime.prolific as prol

    calls = {"n": 0}

    def fake_counts(study_id):
        calls["n"] += 1
        return ({"COMPLETED": 0}, None)

    monkeypatch.setattr(prol, "get_submission_counts", fake_counts)
    completed = collect._poll_prolific_until_target(
        "study", target_places=5, out_dir=tmp_path, max_wait_sec=0, poll_interval_sec=0
    )
    assert completed == 0
    assert calls["n"] == 1  # polled once, then timed out — did not spin


def test_poll_prolific_returns_when_target_met(tmp_path, monkeypatch):
    from src.pipelines.outer_loop import collect
    import src.runtime.prolific as prol

    monkeypatch.setattr(prol, "get_submission_counts", lambda s: ({"COMPLETED": 5}, None))
    completed = collect._poll_prolific_until_target(
        "study", target_places=5, out_dir=tmp_path, max_wait_sec=10_000, poll_interval_sec=0
    )
    assert completed == 5


def _make_exp_dir_no_api(tmp_path):
    import json

    exp_dir = tmp_path / "experiment1"
    (exp_dir / "design").mkdir(parents=True)
    (exp_dir / "design" / "stimuli.json").write_text(
        json.dumps([{"sequence_a": "HT", "sequence_b": "TH"}]), encoding="utf-8"
    )
    (exp_dir / "experiment").mkdir()
    (exp_dir / "experiment" / "config.json").write_text("{}", encoding="utf-8")
    return exp_dir


def test_live_mode_without_results_api_raises_instead_of_synthetic_fallback(tmp_path):
    # --mode live with no deployed experiment must NOT silently fall back to
    # synthetic prior-predictive data; it must fail loudly.
    from src.pipelines.outer_loop import orchestrator

    with pytest.raises(RuntimeError, match="deploy-target"):
        orchestrator.run_collect_programmatic(
            _make_exp_dir_no_api(tmp_path),
            mode="live",
            n_participants=2,
            project_id="subjective_randomness",
        )


def test_empty_participant_collection_raises(tmp_path, monkeypatch):
    from src.pipelines.outer_loop import collect, orchestrator

    monkeypatch.setattr(collect, "_collect_from_firebase", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="no data"):
        orchestrator.run_collect_programmatic(
            _make_exp_dir(tmp_path),
            mode="simulated_participants",
            n_participants=2,
            project_id="subjective_randomness",
        )


def test_malformed_experiment_config_raises(tmp_path):
    from src.pipelines.outer_loop import orchestrator

    exp_dir = _make_exp_dir_no_api(tmp_path)
    (exp_dir / "experiment" / "config.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="[Mm]alformed"):
        orchestrator.run_collect_programmatic(
            exp_dir,
            mode="simulated_participants",
            n_participants=2,
            project_id="subjective_randomness",
        )


def test_programmatic_collect_accepts_varied_firebase_data(tmp_path, monkeypatch):
    import csv

    from src.pipelines.outer_loop import collect, orchestrator

    monkeypatch.setattr(
        collect, "_collect_from_firebase", lambda *a, **k: _rows([1, 0, 1, 0])
    )
    csv_path = orchestrator.run_collect_programmatic(
        _make_exp_dir(tmp_path),
        mode="simulated_participants",
        n_participants=2,
        project_id="subjective_randomness",
    )
    assert csv_path.exists()
    assert len(list(csv.DictReader(csv_path.open(encoding="utf-8")))) == 4
