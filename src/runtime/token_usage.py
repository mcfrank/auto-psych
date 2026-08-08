"""Process-wide accounting of LLM token usage for a pipeline run.

Every LLM spend in the pipeline — coding-agent subprocesses launched via
``run_coding_agent`` and hosted-API calls made via ``invoke_llm`` — reports a
:class:`UsageRecord` here. The log is a single in-memory list for the whole
process (agents may record from worker threads, so all mutation is locked),
optionally mirrored line-by-line to a JSONL sink so a crashed run still keeps
the usage it already paid for.

Token components are DISJOINT: ``input_tokens`` excludes cache reads,
``output_tokens`` excludes reasoning/thinking tokens, and the total is simply
the sum of all five components. Backends report these differently (opencode
already splits them; Anthropic folds reasoning into output; LangChain reports
details as subsets), so the adapters in ``coding_agent`` / ``llm`` normalize
to this convention before recording.

A record with ``usage_missing=True`` means the backend completed without
reporting usage — the call happened and cost tokens, but the count is unknown
(NOT zero). Summaries surface these separately so a total is never silently
an undercount.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class UsageRecord:
    """Token usage of one LLM call (or one coding-agent run)."""

    source: str
    """Pipeline stage that spent the tokens (e.g. ``outer:2_design``)."""
    backend: str
    """What produced the usage numbers: ``claude``, ``opencode``, ``langchain``."""
    model: Optional[str]
    input_tokens: int = 0
    """Non-cached prompt tokens."""
    output_tokens: int = 0
    """Completion tokens, excluding reasoning."""
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Optional[float] = None
    """Backend-reported cost, when available."""
    usage_missing: bool = False
    """True when the backend reported no usage — the spend is unknown, not zero."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


_lock = threading.Lock()
_records: List[UsageRecord] = []
_sink_path: Optional[Path] = None


def start_usage_log(sink_path: Path) -> int:
    """Mirror subsequent records to ``sink_path`` (JSONL, appended).

    Returns a marker for :func:`records_since`, so a caller can later summarize
    exactly the records made after this point (e.g. one experiment's worth).
    """
    global _sink_path
    sink_path = Path(sink_path)
    sink_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _sink_path = sink_path
        return len(_records)


def stop_usage_log() -> None:
    """Detach the JSONL sink; recording continues in memory only."""
    global _sink_path
    with _lock:
        _sink_path = None


def reset_usage_log() -> None:
    """Drop all records and detach the sink (for tests)."""
    global _sink_path
    with _lock:
        _records.clear()
        _sink_path = None


def records_marker() -> int:
    """Current position in the log, for a later :func:`records_since`."""
    with _lock:
        return len(_records)


def records_since(marker: int) -> List[UsageRecord]:
    """All records made after ``marker`` (0 = everything this process)."""
    with _lock:
        return list(_records[marker:])


def record_usage(**fields) -> UsageRecord:
    """Append one :class:`UsageRecord` (and write it to the sink, if any)."""
    record = UsageRecord(**fields)
    with _lock:
        _records.append(record)
        if _sink_path is not None:
            line = json.dumps(asdict(record))
            with _sink_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    return record


def summarize(records: List[UsageRecord]) -> Dict:
    """Totals over ``records``, overall and per source.

    ``cost_usd`` sums only the records that reported a cost; it is ``None``
    when no record did. ``n_calls_missing_usage`` counts calls whose spend is
    unknown — a nonzero value means the token totals are an undercount.
    """

    def _totals(recs: List[UsageRecord]) -> Dict:
        known_costs = [r.cost_usd for r in recs if r.cost_usd is not None]
        return {
            "n_calls": len(recs),
            "n_calls_missing_usage": sum(r.usage_missing for r in recs),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "reasoning_tokens": sum(r.reasoning_tokens for r in recs),
            "cache_read_tokens": sum(r.cache_read_tokens for r in recs),
            "cache_write_tokens": sum(r.cache_write_tokens for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "cost_usd": sum(known_costs) if known_costs else None,
        }

    summary = _totals(records)
    by_source: Dict[str, Dict] = {}
    for source in sorted({r.source for r in records}):
        by_source[source] = _totals([r for r in records if r.source == source])
    summary["by_source"] = by_source
    return summary


def write_usage_report(out_dir: Path, marker: int, *, heading: str) -> Dict:
    """Summarize usage since ``marker``; persist and print it.

    Writes ``token_usage_summary.json`` into ``out_dir`` and prints the
    human-readable summary to the run log. Returns the summary dict.
    """
    summary = summarize(records_since(marker))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_usage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(format_summary(summary, heading), flush=True)
    return summary


def format_summary(summary: Dict, heading: str) -> str:
    """Human-readable multi-line rendering of a :func:`summarize` dict."""
    cost = summary["cost_usd"]
    cost_str = f"  cost=${cost:.2f}" if cost is not None else ""
    missing = summary["n_calls_missing_usage"]
    missing_str = (
        f"  (WARNING: {missing} call(s) reported no usage — totals are an undercount)"
        if missing
        else ""
    )
    lines = [
        f"  [tokens] {heading}: {summary['total_tokens']:,} tokens over "
        f"{summary['n_calls']} LLM call(s){cost_str}{missing_str}",
        f"  [tokens]   input={summary['input_tokens']:,} "
        f"output={summary['output_tokens']:,} "
        f"reasoning={summary['reasoning_tokens']:,} "
        f"cache_read={summary['cache_read_tokens']:,} "
        f"cache_write={summary['cache_write_tokens']:,}",
    ]
    for source, totals in summary["by_source"].items():
        lines.append(
            f"  [tokens]   {source}: {totals['total_tokens']:,} tokens "
            f"({totals['n_calls']} call(s))"
        )
    return "\n".join(lines)
