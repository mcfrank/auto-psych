"""Assemble API payloads from a discovered session plus its live data.

This is the glue between discovery (what to watch), the sources (Firestore +
Prolific), and the pure aggregation in ``aggregate.py``. Keeping it separate
leaves the aggregation math free of any I/O.
"""

from __future__ import annotations

from typing import Any

from src.monitor.aggregate import participant_stat, summarize_choice_balance
from src.monitor.discovery import MonitoredSession
from src.monitor.models import (
    ProlificStatus,
    SessionDetail,
    SessionSummary,
)
from src.monitor.sources import ProlificSource


def _participant_stats(responses: list[tuple[str, dict[str, Any]]]):
    return [participant_stat(doc_id, data) for doc_id, data in responses]


def _summary_fields(session: MonitoredSession, responses) -> dict[str, Any]:
    stats = _participant_stats(responses)
    balance = summarize_choice_balance(stats)
    submitted = [s.submitted_at for s in stats if s.submitted_at]
    n_degenerate = sum(1 for s in stats if s.degenerate)
    return {
        "collection_session_id": session.collection_session_id,
        "experiment_id": session.experiment_id,
        "project_id": session.project_id,
        "prolific_mode": session.prolific_mode,
        "prolific_study_id": session.prolific_study_id,
        "deploy_target": session.deploy_target,
        "experiment_url": session.experiment_url,
        "created_at": session.created_at,
        "target_participants": session.target_participants,
        "n_responses": len(responses),
        "n_with_data": sum(1 for s in stats if s.n_valid_trials > 0),
        "last_submission_at": max(submitted) if submitted else None,
        "overall_p_left": balance.p_left,
        "n_degenerate_participants": n_degenerate,
        "has_warning": balance.is_degenerate or n_degenerate > 0,
    }


def build_summary(session: MonitoredSession, responses) -> SessionSummary:
    return SessionSummary(**_summary_fields(session, responses))


def build_detail(
    session: MonitoredSession,
    responses,
    prolific: ProlificSource | None,
) -> SessionDetail:
    stats = _participant_stats(responses)
    # Newest submission first, so the latest arrivals are easiest to eyeball.
    stats.sort(key=lambda s: s.submitted_at or "", reverse=True)
    balance = summarize_choice_balance(stats)
    return SessionDetail(
        **_summary_fields(session, responses),
        participants=stats,
        choice_balance=balance,
        prolific=fetch_prolific_status(session.prolific_study_id, prolific),
    )


def fetch_prolific_status(
    study_id: str | None,
    prolific: ProlificSource | None,
) -> ProlificStatus:
    """Fetch recruitment status, degrading gracefully when unavailable."""
    empty = ProlificStatus(
        study_id=study_id,
        status=None,
        places_total=None,
        places_taken=None,
        counts={},
        error=None,
    )
    if not study_id or prolific is None:
        return empty

    status_data, status_err = prolific.study_status(study_id)
    counts_data, counts_err = prolific.submission_counts(study_id)

    errors = [e for e in (status_err, counts_err) if e]
    status_data = status_data or {}
    counts = {str(k): int(v) for k, v in (counts_data or {}).items() if isinstance(v, (int, float))}

    return ProlificStatus(
        study_id=study_id,
        status=status_data.get("status"),
        places_total=status_data.get("total_available_places"),
        places_taken=status_data.get("places_taken"),
        counts=counts,
        error="; ".join(errors) if errors else None,
    )
