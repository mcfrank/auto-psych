"""A candidate whose hypothesis repeats an existing model's is rejected (rehearsal 3)."""

from pathlib import Path

import pytest

from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger, LedgerEntry
from src.rsa.loop import orchestrator as orch
from src.rsa.loop.orchestrator import Live, LoopConfig, RSALoop


@pytest.fixture
def loop(tmp_path):
    lp = RSALoop(LoopConfig(responses_path=Path("r"), seed_models_dir=Path("s"), results_dir=tmp_path), lambda d, p: True)
    lp.live = {"parent": Live("parent", "Listeners reason at depth 2 about a confusable speaker.", None, None, None, "seed")}
    lp.ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    lp.ledger.append(LedgerEntry(name="gone", outcome="admitted", detail="", hypothesis="A salience prior.", context="r1"))
    return lp


def test_a_verbatim_hypothesis_is_found_ignoring_whitespace_and_case(loop):
    assert loop._same_hypothesis("listeners reason at depth 2\nabout a confusable   speaker.") == "parent"
    assert loop._same_hypothesis("A SALIENCE PRIOR.") == "gone"  # admitted earlier, since retired
    assert loop._same_hypothesis("Listeners reason at depth 2 about a confusable speaker, with a cost.") is None
    assert loop._same_hypothesis("  ") is None


def test_the_candidate_is_rejected_with_the_reason_before_any_fit(loop, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "admit", lambda *a, **k: pytest.fail("admission must not fit a copied hypothesis"))
    cdir = tmp_path / "candidate_1"
    cdir.mkdir()
    (cdir / "candidate.py").write_text("# a changed model\n")
    (cdir / "hypothesis.md").write_text("Listeners reason at depth 2 about a confusable speaker.\n")
    ok, reason = loop._try_admit(cdir, "child", "round 1 candidate 1 refine incumbent parent")
    assert not ok and "word for word parent's hypothesis" in reason
    last = loop.ledger.entries()[-1]
    assert (last.name, last.outcome) == ("child", "rejected") and "word for word" in last.detail
