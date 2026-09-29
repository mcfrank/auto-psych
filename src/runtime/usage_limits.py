"""Recognise an agent CLI's usage or rate limit, and when it lifts.

A coding agent that runs into its account's limit ends at once, and the CLI
says so in its output rather than failing in a way the harness can tell apart
from an agent that wrote nothing:

- Claude, subscription login: the terminal ``result`` event (``subtype
  "success"``) carries ``You've hit your session limit · resets 2:20pm
  (America/Los_Angeles)`` as its text, also after the agent had worked for a
  while. Older CLIs wrote ``Claude AI usage limit reached|<unix time>``.
- Claude, API key: after its own retries the CLI ends with ``API Error: 429
  {... "rate_limit_error" ...}``, ``API Error: 529 {... "overloaded_error"
  ...}`` or ``API Error: Repeated 529 Overloaded errors``.
- opencode (Gemini and other providers): an ``error`` event whose error
  carries the provider's message (Gemini: ``RESOURCE_EXHAUSTED``, "You
  exceeded your current quota", "Resource has been exhausted"; HTTP 429 or a
  503 "model is overloaded").
- codex: an ``error`` or ``turn.failed`` event, e.g. "You've hit your usage
  limit ... try again in 2 hours 5 minutes" or "exceeded retry limit, last
  status: 429 Too Many Requests".

The harness (``coding_agent.run_coding_agent``) treats such a run as an
infrastructure condition: it waits until the stated reset and runs the agent
again. Only the CLI's own channels are read — Claude's result text (its first
``LIMIT_WINDOW`` characters), opencode's and codex's error events, and lines
the CLI printed outside its JSON stream — never the agent's tool calls or
their output, so an agent that reads this file does not look limited.

The opencode and codex texts are those the providers document; no run of this
pipeline has recorded one yet.

Some conditions no wait cures (an exhausted API credit balance, a rejected
key): :func:`detect_login_failure` finds those, and the launcher raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

# Claude: "You've hit your session limit · resets 12am (America/Los_Angeles)";
# other CLIs phrase it as a usage/rate/weekly limit.
LIMIT_RE = re.compile(
    r"(?:(?:hit|reached|exceeded) (?:your |the )?(?:session |usage |weekly |rate |plan )?limit"
    r"|(?:usage|rate|session|weekly|plan) limit (?:reached|exceeded|hit)"
    r"|usage_limit)",
    re.IGNORECASE,
)
# The API's rate-limit and overload responses as the CLIs print them.
RATE_LIMIT_RE = re.compile(
    r"API Error: (?:429|529)|Repeated 529|rate_limit_error|overloaded_error"
    r"|RESOURCE_EXHAUSTED|exceeded your current quota|Resource has been exhausted"
    r"|Too Many Requests|model is overloaded|\"statusCode\":\s*(?:429|529)",
    re.IGNORECASE,
)
# Claude's result text is the agent's own final message when it did not hit a
# limit: a limit message is matched only at its start, so a message that
# merely discusses limits does not trip it.
LIMIT_WINDOW = 200
# "resets 12am (America/Los_Angeles)", "resets 3:30pm", "resets at 9 pm",
# "try again at 3:00 PM"
_CLOCK_RE = re.compile(
    r"(?:resets?|try again|available again|retry)\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b"
    r"(?:\s*\(([^)]+)\))?",
    re.IGNORECASE,
)
# "resets in 2 hours", "try again in 4h 30m", "resets in 45 minutes",
# "available again in 1 hour and 5 minutes", "try again in 3 days 1 hour"
_RELATIVE_RE = re.compile(
    r"(?:resets?|try again|available again|retry)\s+(?:in|after)\s+"
    r"(?:(\d+)\s*(?:d\b|day)s?,?\s*)?"
    r"(?:(\d+)\s*(?:h\b|hr|hour)s?)?,?\s*(?:and\s*)?(?:(\d+)\s*(?:m\b|min|minute)s?)?",
    re.IGNORECASE,
)
# "Claude AI usage limit reached|1759852800" (older Claude CLIs).
_EPOCH_RE = re.compile(r"\|(\d{10})\b")
# A clock time this far in the past is the reset that has just happened (the
# message was written a moment before it), not tomorrow's.
RESET_JUST_PASSED = timedelta(minutes=15)

# No wait cures these: the account cannot pay for the call, or the key is bad.
_LOGIN_FAILURE_RE = re.compile(
    r"Credit balance is too low|Invalid API key", re.IGNORECASE
)


@dataclass(frozen=True)
class UsageLimit:
    message: str
    reset_at: Optional[datetime]
    """Timezone-aware reset time, or None if the message named none."""


class AgentInfrastructureError(RuntimeError):
    """The agent CLI could not work for a reason that is not the agent's:
    never a candidate's failure, never a ledger line; it ends the run."""


class AgentUsageLimitExceeded(AgentInfrastructureError):
    """An agent's usage or rate limit did not lift within the maximum wait."""


class AgentLoginFailed(AgentInfrastructureError):
    """The agent's login cannot make calls (no credit, a rejected key)."""


def parse_reset_time(text: str, now: datetime) -> Optional[datetime]:
    """The reset instant named in a limit message, or None. ``now`` must be aware."""
    match = _CLOCK_RE.search(text)
    if match:
        hour, minute, meridiem, zone = match.groups()
        hour = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
        tz = ZoneInfo(zone) if zone else now.tzinfo
        local_now = now.astimezone(tz)
        reset = local_now.replace(hour=hour, minute=int(minute or 0), second=0, microsecond=0)
        if reset <= local_now - RESET_JUST_PASSED:
            reset += timedelta(days=1)
        return reset
    match = _RELATIVE_RE.search(text)
    if match and any(match.groups()):
        days, hours, minutes = (int(group or 0) for group in match.groups())
        return now + timedelta(days=days, hours=hours, minutes=minutes)
    match = _EPOCH_RE.search(text)
    if match:
        return datetime.fromtimestamp(int(match.group(1)), tz=timezone.utc)
    return None


def detect_usage_limit(
    messages: Iterable[str], now: Optional[datetime] = None
) -> Optional[UsageLimit]:
    """The first of ``messages`` (texts from the CLI's own channels) that is a
    usage- or rate-limit message, as a :class:`UsageLimit`; else None."""
    for message in messages:
        if message and (LIMIT_RE.search(message) or RATE_LIMIT_RE.search(message)):
            now = now or datetime.now().astimezone()
            return UsageLimit(message=message.strip(), reset_at=parse_reset_time(message, now))
    return None


def detect_login_failure(messages: Iterable[str]) -> Optional[str]:
    """The first of ``messages`` that says the login cannot make calls, or None."""
    for message in messages:
        if message and _LOGIN_FAILURE_RE.search(message):
            return message.strip()
    return None


def seconds_until_retry(
    limit: UsageLimit, now: datetime, *, margin_sec: float, fallback_sec: float
) -> float:
    """How long to wait before running the agent again: until ``margin_sec``
    after the stated reset, or ``fallback_sec`` when the message named none."""
    if limit.reset_at is None:
        return float(fallback_sec)
    return max(0.0, (limit.reset_at - now).total_seconds()) + float(margin_sec)
