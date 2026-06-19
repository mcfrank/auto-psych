"""Pydantic payloads returned by the monitor API.

These mirror exactly what the browser consumes, so the frontend and the Python
aggregation never drift. Choice statistics are centered on the binary
``chose_left`` response used by the subjective-randomness task; trials without a
choice are still counted (as trials) but do not contribute to choice balance.
"""

from __future__ import annotations

from pydantic import BaseModel


class ParticipantStat(BaseModel):
    """One participant's submission, summarized."""

    participant_id: str
    prolific_pid: str | None
    n_trials: int
    n_valid_trials: int
    n_left: int
    n_right: int
    p_left: float | None
    submitted_at: str | None
    degenerate: bool


class ChoiceBalance(BaseModel):
    """Aggregate left/right split across all valid trials in a session."""

    total_valid_trials: int
    n_left: int
    n_right: int
    p_left: float | None
    is_degenerate: bool
    warning: str | None


class ProlificStatus(BaseModel):
    """Recruitment status for a session's Prolific study, if it has one."""

    study_id: str | None
    status: str | None
    places_total: int | None
    places_taken: int | None
    counts: dict[str, int]
    error: str | None


class SessionSummary(BaseModel):
    """A live study at a glance — one card in the dashboard list."""

    collection_session_id: str
    experiment_id: str
    project_id: str
    prolific_mode: str
    prolific_study_id: str | None
    deploy_target: str
    experiment_url: str | None
    created_at: str | None
    target_participants: int | None
    n_responses: int
    n_with_data: int
    last_submission_at: str | None
    overall_p_left: float | None
    n_degenerate_participants: int
    has_warning: bool


class SessionDetail(SessionSummary):
    """Everything in the summary plus the per-participant and recruitment detail."""

    participants: list[ParticipantStat]
    choice_balance: ChoiceBalance
    prolific: ProlificStatus
