"""A round where every candidate slot yields no file must raise.

When 292 of 297 candidate rejections are "no candidate.py written", the loop
proceeds as if the agents simply declined every round — and the final metric
measures a loop that could barely propose models. A loud failure after such a
round catches the configuration bug (a missing opencode ``write`` permission,
a cwd outside the worktree) before it wastes a 20-cell sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.inner_loop.model_zoo import (
    AllCandidatesNoFileError,
    _check_round_admissions,
)


def test_all_no_file_raises():
    """Zero admitted, all rejections are 'no candidate.py written' => error."""
    round_results = [
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    with pytest.raises(AllCandidatesNoFileError, match="no candidate.py written"):
        _check_round_admissions(round_results, round_context="round 0")


def test_some_admitted_does_not_raise():
    """At least one admitted => no error."""
    round_results = [
        {"outcome": "admitted", "detail": ""},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    _check_round_admissions(round_results, round_context="round 0")


def test_rejected_for_other_reason_does_not_raise():
    """Zero admitted but some rejections are for a real reason => no error.

    When the agent wrote a candidate.py but it failed to load, fit, or pass the
    novelty gate, the loop is working as designed — the agent tried, the model
    was bad. That is not a configuration bug; it is a bad candidate.
    """
    round_results = [
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "rejected", "detail": "candidate.py is not a loadable PyMC model: ..."},
        {"outcome": "rejected", "detail": "no candidate.py written"},
    ]
    _check_round_admissions(round_results, round_context="round 0")


def test_spawn_failure_counted_as_no_file():
    """When the agent process itself failed (spawn_ok=False), admission was never
    attempted: that slot produced no file. If ALL slots are spawn failures or
    no-file rejections, it should raise.
    """
    round_results = [
        {"outcome": "spawn_failed", "detail": "agent process failed"},
        {"outcome": "rejected", "detail": "no candidate.py written"},
        {"outcome": "spawn_failed", "detail": "agent process failed"},
    ]
    with pytest.raises(AllCandidatesNoFileError):
        _check_round_admissions(round_results, round_context="round 0")


def test_empty_round_is_valid():
    """A round with candidate_count=0 is valid (fit-only, no agents spawned)."""
    _check_round_admissions([], round_context="round 0")
