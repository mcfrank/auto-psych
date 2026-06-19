"""Unit tests for monitor aggregation — the data-quality math.

The aggregation turns raw Firestore response docs into per-participant and
per-session statistics, with explicit detection of the degenerate-data failure
mode (everyone choosing the same side) that must never pass unnoticed again.
"""

from __future__ import annotations

import pytest

from src.monitor.aggregate import (
    participant_stat,
    summarize_choice_balance,
)
from tests.monitor_fixtures import response_doc


def test_participant_stat_counts_choices():
    doc_id, data = response_doc("PID_A", [True, False, True, False, True], created_at="2026-06-19T18:05:00Z")
    stat = participant_stat(doc_id, data)
    assert stat.participant_id == "PID_A"
    assert stat.n_trials == 5
    assert stat.n_valid_trials == 5
    assert stat.n_left == 3
    assert stat.n_right == 2
    assert stat.p_left == pytest.approx(0.6)
    assert stat.submitted_at == "2026-06-19T18:05:00Z"
    assert stat.degenerate is False


def test_participant_stat_flags_all_one_side():
    doc_id, data = response_doc("PID_C", [True] * 6, created_at="2026-06-19T19:05:00Z")
    stat = participant_stat(doc_id, data)
    assert stat.n_left == 6
    assert stat.n_right == 0
    assert stat.p_left == pytest.approx(1.0)
    assert stat.degenerate is True


def test_participant_stat_ignores_invalid_trials():
    data = {
        "prolific_pid": "PID_X",
        "created_at": "2026-06-19T18:00:00Z",
        "trials": [
            {"sequence_a": [0, 1], "sequence_b": [1, 1], "chose_left": True},
            {"sequence_a": None, "sequence_b": [1, 1], "chose_left": False},  # no stimulus
            {"sequence_a": [0, 1], "sequence_b": [1, 1], "chose_left": None},  # no response
        ],
    }
    stat = participant_stat("PID_X", data)
    assert stat.n_trials == 3
    assert stat.n_valid_trials == 1
    assert stat.n_left == 1


def test_too_few_trials_is_not_degenerate():
    # A single one-sided trial is not enough evidence to call a participant bad.
    doc_id, data = response_doc("PID_Y", [True], created_at="2026-06-19T18:00:00Z")
    stat = participant_stat(doc_id, data)
    assert stat.degenerate is False


def test_choice_balance_healthy():
    stats = [
        participant_stat(*response_doc("A", [True, False, True, False], created_at="t")),
        participant_stat(*response_doc("B", [False, True, False, True], created_at="t")),
    ]
    balance = summarize_choice_balance(stats)
    assert balance.total_valid_trials == 8
    assert balance.p_left == pytest.approx(0.5)
    assert balance.is_degenerate is False
    assert balance.warning is None


def test_choice_balance_degenerate_all_left():
    stats = [participant_stat(*response_doc(f"P{i}", [True] * 6, created_at="t")) for i in range(3)]
    balance = summarize_choice_balance(stats)
    assert balance.p_left == pytest.approx(1.0)
    assert balance.is_degenerate is True
    assert "left" in balance.warning.lower()


def test_choice_balance_empty_is_not_degenerate():
    balance = summarize_choice_balance([])
    assert balance.total_valid_trials == 0
    assert balance.p_left is None
    assert balance.is_degenerate is False
