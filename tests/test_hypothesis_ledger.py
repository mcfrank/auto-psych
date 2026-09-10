"""Unit tests for the hypothesis ledger (``attempted_hypotheses.jsonl``).

One JSON line per event — a candidate admitted or rejected, a model pruned or
dropped — appended as it happens so a crashed run still leaves the record. The
ledger starts from the one the previous experiment carried, and renders the
retired hypotheses (those no longer in the model set) as the "already tried"
section of the candidate brief.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
    LedgerEntry,
    one_line,
)


def _entry(name, outcome, detail="", hypothesis="People do X.", context="round 0"):
    return LedgerEntry(
        name=name, outcome=outcome, detail=detail, hypothesis=hypothesis, context=context
    )


def test_create_starts_empty_without_an_inherited_ledger(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    assert ledger.path.exists()
    assert ledger.entries() == []


def test_create_copies_the_inherited_ledger_and_appends_after_it(tmp_path):
    inherited = tmp_path / "prev" / LEDGER_FILENAME
    inherited.parent.mkdir()
    inherited.write_text(_entry("old", "pruned").to_json() + "\n", encoding="utf-8")

    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=inherited)
    ledger.append(_entry("new", "admitted"))

    assert [(e.name, e.outcome) for e in ledger.entries()] == [
        ("old", "pruned"),
        ("new", "admitted"),
    ]
    # The inherited file itself is untouched.
    assert inherited.read_text(encoding="utf-8").count("\n") == 1


def test_create_ignores_an_absent_inherited_path(tmp_path):
    ledger = HypothesisLedger.create(
        tmp_path / LEDGER_FILENAME, inherit_from=tmp_path / "nowhere" / LEDGER_FILENAME
    )
    assert ledger.entries() == []


def test_create_overwrites_a_stale_ledger_at_the_target(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    path.write_text(_entry("stale", "admitted").to_json() + "\n", encoding="utf-8")
    ledger = HypothesisLedger.create(path, inherit_from=None)
    assert ledger.entries() == []


def test_entries_round_trip_every_field(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    entry = _entry("m", "rejected", detail="near-duplicate of a", context="round 2 candidate 1")
    ledger.append(entry)
    assert ledger.entries() == [entry]
    assert json.loads(ledger.path.read_text(encoding="utf-8"))["name"] == "m"


def test_unknown_outcome_is_refused():
    with pytest.raises(ValueError, match="outcome"):
        _entry("m", "vanished")


def test_malformed_line_fails_loudly(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attempted_hypotheses"):
        HypothesisLedger(path).entries()


def test_missing_ledger_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        HypothesisLedger(tmp_path / LEDGER_FILENAME).entries()


def test_retired_is_the_latest_outcome_per_name_not_in_the_live_set(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    ledger.append(_entry("a", "admitted"))
    ledger.append(_entry("b", "admitted"))
    ledger.append(_entry("a", "pruned", detail="20 nats behind seed"))
    ledger.append(_entry("c", "rejected", detail="no candidate.py"))

    retired = ledger.retired(live_names={"b"})

    assert [(e.name, e.outcome, e.detail) for e in retired] == [
        ("a", "pruned", "20 nats behind seed"),
        ("c", "rejected", "no candidate.py"),
    ]


def test_render_lists_retired_hypotheses_under_the_do_not_re_propose_heading(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    ledger.append(
        _entry("runs", "pruned", detail="20.0 nats behind seed (4.0× dse)",
               hypothesis="People dislike long runs.", context="experiment1 round 0")
    )
    ledger.append(_entry("live", "admitted"))

    text = ledger.render_markdown(live_names={"live"})

    assert text.startswith("# Already tried")
    assert "do not re-propose" in text.lower()
    assert "| runs | pruned (experiment1 round 0): 20.0 nats behind seed (4.0× dse) | People dislike long runs. |" in text
    assert "live" not in text.split("|", 1)[1]


def test_render_with_nothing_retired_says_so(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    ledger.append(_entry("live", "admitted"))
    text = ledger.render_markdown(live_names={"live"})
    assert "No earlier hypothesis has been retired yet" in text


def test_one_line_collapses_whitespace_and_truncates():
    assert one_line("People  judge\nby runs.  Then more.") == "People judge by runs. Then more."
    long = "word " * 100
    shortened = one_line(long, limit=40)
    assert len(shortened) <= 41 and shortened.endswith("…")
    assert one_line("   ") == ""
