"""Tests for the participant completion-time analysis script.

``compute_completion_time`` pulls each study's submissions from the Prolific
API (via ``src.runtime.prolific.list_submissions``), keeps the ones in a
"finished" status, and summarizes their ``time_taken`` (seconds Prolific
recorded between a participant starting and submitting the study).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "analysis"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "compute_completion_time", SCRIPTS / "compute_completion_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _submission(status, time_taken, participant="p"):
    return {
        "id": f"sub_{participant}",
        "participant_id": participant,
        "status": status,
        "time_taken": time_taken,
    }


# --- completion_time_statistics ------------------------------------------


def test_completion_time_statistics_are_correct():
    stats = cli.completion_time_statistics([60.0, 120.0, 180.0])
    assert stats["n"] == 3
    assert stats["mean_seconds"] == pytest.approx(120.0)
    assert stats["median_seconds"] == pytest.approx(120.0)
    assert stats["std_seconds"] == pytest.approx(60.0)  # sample stdev
    assert stats["min_seconds"] == pytest.approx(60.0)
    assert stats["max_seconds"] == pytest.approx(180.0)
    assert stats["mean_minutes"] == pytest.approx(2.0)


def test_completion_time_statistics_empty_fails_loudly():
    with pytest.raises(ValueError):
        cli.completion_time_statistics([])


# --- time_taken_for_study (status filtering + loud failure) --------------


def test_time_taken_for_study_keeps_only_finished_statuses(monkeypatch):
    subs = [
        _submission("APPROVED", 100, "a"),
        _submission("AWAITING REVIEW", 200, "b"),
        _submission("RETURNED", 999, "c"),
        _submission("TIMED-OUT", None, "d"),
    ]
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: (subs, None))
    rows = cli.time_taken_for_study("study1", cli.DEFAULT_STATUSES)
    assert [r.seconds for r in rows] == [100.0, 200.0]
    assert [r.participant_id for r in rows] == ["a", "b"]


def test_time_taken_for_study_raises_when_finished_submission_lacks_time(monkeypatch):
    subs = [_submission("APPROVED", None, "a")]
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: (subs, None))
    with pytest.raises(ValueError, match="time_taken"):
        cli.time_taken_for_study("study1", cli.DEFAULT_STATUSES)


def test_time_taken_for_study_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: (None, "boom"))
    with pytest.raises(RuntimeError, match="boom"):
        cli.time_taken_for_study("study1", cli.DEFAULT_STATUSES)


# --- summarize across studies (integration) ------------------------------


def test_summarize_pools_across_studies_and_breaks_down_per_study(monkeypatch):
    by_study = {
        "studyA": [_submission("APPROVED", 60, "a"), _submission("APPROVED", 120, "b")],
        "studyB": [_submission("AWAITING REVIEW", 240, "c")],
    }
    monkeypatch.setattr(
        cli, "list_submissions", lambda study_id: (by_study[study_id], None)
    )
    result = cli.summarize(
        study_ids=("studyA", "studyB"),
        labels=("run1/e1", "run1/e2"),
        statuses=cli.DEFAULT_STATUSES,
    )
    # Overall pools all three participants: mean of 60, 120, 240 = 140s.
    assert result["overall"]["n"] == 3
    assert result["overall"]["mean_seconds"] == pytest.approx(140.0)
    # Per-study breakdown is keyed by the supplied labels.
    assert result["per_study"]["run1/e1"]["n"] == 2
    assert result["per_study"]["run1/e1"]["mean_seconds"] == pytest.approx(90.0)
    assert result["per_study"]["run1/e2"]["n"] == 1


def test_main_prints_mean_completion_time(monkeypatch, capsys):
    subs = [_submission("APPROVED", 300, "a"), _submission("APPROVED", 300, "b")]
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: (subs, None))
    cli.main(cli.Args(study_ids=("studyA",)))
    out = capsys.readouterr().out
    assert "mean" in out.lower()
    assert "5.0" in out  # 300s = 5.0 minutes


def test_main_writes_per_participant_csv(monkeypatch, tmp_path):
    subs = [_submission("APPROVED", 60, "a"), _submission("APPROVED", 180, "b")]
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: (subs, None))
    out_csv = tmp_path / "completion_times.csv"
    cli.main(cli.Args(study_ids=("studyA",), labels=("run1/e1",), output_csv=out_csv))
    text = out_csv.read_text(encoding="utf-8")
    assert "participant_id" in text
    assert "time_taken_seconds" in text
    assert "60" in text and "180" in text


def test_main_rejects_mismatched_labels(monkeypatch):
    monkeypatch.setattr(cli, "list_submissions", lambda study_id: ([], None))
    with pytest.raises(ValueError, match="labels"):
        cli.main(cli.Args(study_ids=("studyA", "studyB"), labels=("only-one",)))
