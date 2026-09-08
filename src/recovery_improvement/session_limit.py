"""Detect a Claude subscription session limit and work out when to retry.

A headless ``claude -p`` session that runs into the subscription's rolling
usage window ends with a ``result`` event whose text reads e.g.
``You've hit your session limit · resets 12am (America/Los_Angeles)``. The
run is not a failure of the agent's work — its edits and transcript are
intact — so the review job requeues itself to start just after the reset and
resumes the iteration then. (Sherlock forbids a job that idles for hours, so
we exit and come back rather than sleep.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

_LIMIT_RE = re.compile(r"(hit|reached) your (session |usage )?limit", re.IGNORECASE)
# "resets 12am (America/Los_Angeles)", "resets 3:30pm", "resets at 9 pm"
_CLOCK_RE = re.compile(
    r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b(?:\s*\(([^)]+)\))?",
    re.IGNORECASE,
)
# "resets in 2 hours", "resets in 45 minutes"
_RELATIVE_RE = re.compile(r"resets?\s+in\s+(\d+)\s*(hour|hr|minute|min)s?", re.IGNORECASE)

RETRY_MARGIN = timedelta(minutes=5)
"""Start this long after the stated reset, so the window has really turned."""
FALLBACK_WAIT = timedelta(hours=1)
"""Retry interval when the message names no reset time we can parse."""


@dataclass(frozen=True)
class SessionLimit:
    message: str
    reset_at: Optional[datetime]
    """Timezone-aware reset time, or None if the message named none."""


class SessionLimitHit(RuntimeError):
    """Raised by the iteration when the review agent hit the session limit."""

    def __init__(self, limit: SessionLimit):
        super().__init__(limit.message)
        self.limit = limit


def parse_reset_time(text: str, now: datetime) -> Optional[datetime]:
    """The reset instant named in a limit message, or None. ``now`` must be aware."""
    match = _CLOCK_RE.search(text)
    if match:
        hour, minute, meridiem, zone = match.groups()
        hour = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
        tz = ZoneInfo(zone) if zone else now.tzinfo
        local_now = now.astimezone(tz)
        reset = local_now.replace(hour=hour, minute=int(minute or 0), second=0, microsecond=0)
        if reset <= local_now:
            reset += timedelta(days=1)
        return reset
    match = _RELATIVE_RE.search(text)
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        delta = timedelta(hours=amount) if unit.startswith("h") else timedelta(minutes=amount)
        return now + delta
    return None


def detect_session_limit(result_text: str, now: Optional[datetime] = None) -> Optional[SessionLimit]:
    """A :class:`SessionLimit` if ``result_text`` is a limit message, else None."""
    if not _LIMIT_RE.search(result_text or ""):
        return None
    now = now or datetime.now().astimezone()
    return SessionLimit(message=result_text.strip(), reset_at=parse_reset_time(result_text, now))


def retry_begin_time(limit: SessionLimit, now: Optional[datetime] = None) -> str:
    """``sbatch --begin`` value (local time, ``YYYY-MM-DDTHH:MM:SS``) for the retry."""
    now = now or datetime.now().astimezone()
    start = (limit.reset_at + RETRY_MARGIN) if limit.reset_at else (now + FALLBACK_WAIT)
    return start.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
