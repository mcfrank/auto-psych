"""A round where every candidate slot yields no file is detected.

``_is_all_no_file_round`` returns True when every slot is a no-file rejection
or spawn failure. The orchestrator uses this to drive the retry and abandon
logic (see ``test_empty_round_retry.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.inner_loop.model_zoo import (
    _is_all_no_file_round,
)


def test_all_no_file_returns_true():
    """Zero admitted, all rejections are 'no candidate.py written' => True."""
    round_results = [
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    assert _is_all_no_file_round(round_results) is True


def test_some_admitted_returns_false():
    """At least one admitted => False."""
    round_results = [
        {"outcome": "admitted", "detail": ""},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    assert _is_all_no_file_round(round_results) is False


def test_rejected_for_other_reason_returns_false():
    """Zero admitted but some rejections are for a real reason => False.

    When the agent wrote a candidate.py but it failed to load, fit, or pass the
    novelty gate, the loop is working as designed — the agent tried, the model
    was bad. That is not a configuration bug; it is a bad candidate.
    """
    round_results = [
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "candidate.py is not a loadable PyMC model: ..."},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    assert _is_all_no_file_round(round_results) is False


def test_spawn_failure_counted_as_no_file():
    """When the agent process itself failed (spawn_ok=False), admission was never
    attempted: that slot produced no file. If ALL slots are spawn failures or
    no-file rejections, it should return True.
    """
    round_results = [
        {"outcome": "spawn_failed", "detail": "agent process failed"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "spawn_failed", "detail": "agent process failed"},
    ]
    assert _is_all_no_file_round(round_results) is True


def test_empty_round_returns_false():
    """A round with candidate_count=0 is valid (fit-only, no agents spawned)."""
    assert _is_all_no_file_round([]) is False
