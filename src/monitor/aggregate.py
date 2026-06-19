"""Turn raw Firestore response docs into monitoring statistics.

A *valid* trial is one that actually carries a stimulus pair and a recorded
choice; partially-filled trials (e.g. a stimulus shown but no answer logged)
are counted toward the trial total but excluded from choice statistics.

The degenerate-data detection here is deliberately explicit. An earlier pilot
was ruined when every simulated participant chose the same side and nobody
noticed until analysis. These thresholds make that loud.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.monitor.models import ChoiceBalance, ParticipantStat

# A participant needs at least this many valid trials before an all-one-side
# pattern is treated as a quality problem (one or two trials prove nothing).
DEGENERATE_PARTICIPANT_MIN_TRIALS = 3

# A session needs at least this many valid trials before an extreme overall
# split is treated as degenerate, so a tiny early sample is not over-flagged.
DEGENERATE_SESSION_MIN_TRIALS = 10

# p_left at or beyond this distance from the extremes (<=0.05 or >=0.95) counts
# as degenerate for a session — essentially everyone picking the same side.
EXTREME_P_LEFT = 0.05


def _iso(value: Any) -> str | None:
    """Coerce a Firestore timestamp (str or datetime) to an ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    # Firestore's DatetimeWithNanoseconds and similar expose isoformat().
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _trial_is_valid(trial: dict[str, Any]) -> bool:
    return (
        trial.get("sequence_a") is not None
        and trial.get("sequence_b") is not None
        and trial.get("chose_left") is not None
    )


def participant_stat(doc_id: str, data: dict[str, Any]) -> ParticipantStat:
    """Summarize one participant's response document."""
    trials = data.get("trials") or []
    valid = [t for t in trials if _trial_is_valid(t)]
    n_left = sum(1 for t in valid if bool(t["chose_left"]))
    n_valid = len(valid)
    n_right = n_valid - n_left
    p_left = (n_left / n_valid) if n_valid else None

    degenerate = (
        n_valid >= DEGENERATE_PARTICIPANT_MIN_TRIALS and (n_left == 0 or n_right == 0)
    )

    submitted_at = _iso(data.get("created_at")) or _iso(data.get("submitted_at_client"))

    return ParticipantStat(
        participant_id=doc_id,
        prolific_pid=data.get("prolific_pid"),
        n_trials=len(trials),
        n_valid_trials=n_valid,
        n_left=n_left,
        n_right=n_right,
        p_left=p_left,
        submitted_at=submitted_at,
        degenerate=degenerate,
    )


def summarize_choice_balance(stats: list[ParticipantStat]) -> ChoiceBalance:
    """Aggregate the left/right split across all participants in a session."""
    n_left = sum(s.n_left for s in stats)
    n_right = sum(s.n_right for s in stats)
    total = n_left + n_right
    p_left = (n_left / total) if total else None

    is_degenerate = False
    warning: str | None = None
    if total >= DEGENERATE_SESSION_MIN_TRIALS and p_left is not None:
        if p_left <= EXTREME_P_LEFT:
            is_degenerate = True
            warning = (
                f"{p_left:.0%} of choices went left across {total} trials — "
                "participants are choosing almost entirely one side (right). "
                "Check the task and response mapping before collecting more."
            )
        elif p_left >= 1 - EXTREME_P_LEFT:
            is_degenerate = True
            warning = (
                f"{p_left:.0%} of choices went left across {total} trials — "
                "participants are choosing almost entirely one side (left). "
                "Check the task and response mapping before collecting more."
            )

    return ChoiceBalance(
        total_valid_trials=total,
        n_left=n_left,
        n_right=n_right,
        p_left=p_left,
        is_degenerate=is_degenerate,
        warning=warning,
    )
