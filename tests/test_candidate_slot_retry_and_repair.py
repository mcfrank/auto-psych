"""No candidate slot is lost silently: empty slots are retried once, rejected
candidates are repaired once, and every attempt is in the ledger.

In the 2026-09 sweep, 104 of 360 candidate slots ended as "no candidate.py
written" and a round was only retried when *every* slot was empty (~2% of
three-slot rounds; it never fired). The 29 rejections that did carry a reason
("predicts like existing model X", "not a loadable PyMC model", ...) were
never shown to the agent, which had no second attempt. Now:

- a slot whose agent wrote no ``candidate.py`` is re-spawned once, in its own
  directory (``candidate_<i>_retry_1``), and that is final if it stays empty;
- a candidate rejected at admission is re-spawned once, in its own directory
  (``candidate_<i>_repair_1``), with the rejection reason injected verbatim
  into the prompt and the rejected files copied in as a starting point; a
  second rejection is final;
- the all-slots-empty round retry stays as the outer guard.

MCMC, agents and scoring are stubbed; only the slot bookkeeping runs.
"""

from __future__ import annotations

import json

import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import write_responses, write_seed_models


def _manifest_names(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return [e["name"] for e in data["models"]]


def _patch_scoring(monkeypatch, *, near_duplicate_names=()):
    """``model_a`` always wins; nothing is ever pruned.

    A candidate whose admitted name is in ``near_duplicate_names`` predicts
    like ``model_a`` (RMSE 0.001) and is rejected by the novelty gate; every
    other candidate is novel.
    """

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        return {
            "posteriors": {n: (1.0 if n == "model_a" else 0.0) for n in names},
            "elpd_loo": {n: (-10.0 if n == "model_a" else -11.0) for n in names},
            "n_trials": 2,
        }

    def fake_compare(responses_path, models_dir, **kwargs):
        rows = {}
        for rank, n in enumerate(_manifest_names(models_dir)):
            rows[n] = {
                "rank": rank,
                "elpd_loo": -10.0 - rank,
                "elpd_diff": 0.0 if n == "model_a" else 1.0,
                "dse": 0.0 if n == "model_a" else 5.0,
                "dse_clustered": 0.0 if n == "model_a" else 5.0,
                "weight": 1.0 if n == "model_a" else 0.0,
                "loo_unreliable": False,
            }
        return rows

    def fake_min_rmse(name, *a, **k):
        if name in near_duplicate_names:
            return "model_a", 0.001
        return None, float("inf")

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    # The stub fit is not a real trace: pass the convergence gate.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, models_dir: object())
    monkeypatch.setattr(model_zoo, "_min_prediction_rmse", fake_min_rmse)


def _write_candidate(candidate_dir, *, name, hypothesis="People use H."):
    (candidate_dir / "candidate.py").write_text(f"# {name}\n", encoding="utf-8")
    (candidate_dir / "hypothesis.md").write_text(hypothesis + "\n", encoding="utf-8")
    (candidate_dir / "model_name.txt").write_text(name + "\n", encoding="utf-8")


def _ledger_rows(results_dir):
    path = results_dir / LEDGER_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path, monkeypatch, fake_spawn, *, candidate_count):
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"
    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=candidate_count,
        enable_critique=False,
        ledger_context="experiment1",
    )
    return results_dir


# ── Per-slot retry after an empty attempt ─────────────────────────────


def test_empty_slot_is_retried_once_in_its_own_directory(tmp_path, monkeypatch):
    """Slot 1 writes nothing on its first attempt and a full candidate on its
    retry; slot 0 is fine throughout. Only slot 1 is re-spawned, once, in
    ``iter_0/candidate_1_retry_1``, and both of its attempts are in the ledger."""
    _patch_scoring(monkeypatch)
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append((candidate_dir, docs))
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="first_idea")
        elif candidate_dir.name == "candidate_1_retry_1":
            _write_candidate(candidate_dir, name="second_idea")
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=2)

    assert [d.name for d, _ in spawns] == [
        "candidate_0",
        "candidate_1",
        "candidate_1_retry_1",
    ]
    retry_dir, retry_docs = spawns[2]
    assert retry_dir == results_dir / "iter_0" / "candidate_1_retry_1"
    # The retry gets a fresh, complete context of its own …
    assert (retry_dir / "CONTEXT.md").exists()
    assert (retry_dir / "CANDIDATE_BRIEF.md").exists()
    # … and is told that it is a retry after an attempt that wrote nothing.
    assert retry_docs["attempt_note"]
    assert "candidate.py" in retry_docs["attempt_note"]
    assert str(results_dir / "iter_0" / "candidate_1") in retry_docs["attempt_note"]
    assert (retry_dir / "ATTEMPT_NOTE.md").read_text(encoding="utf-8") == retry_docs[
        "attempt_note"
    ]
    # The first attempt of a slot carries no note.
    assert spawns[0][1]["attempt_note"] is None

    assert _manifest_names(results_dir / "models") == [
        "model_a",
        "model_b",
        "first_idea",
        "second_idea",
    ]
    # Slot 1 of a two-slot round is the incumbent-refinement slot; its
    # context names the incumbent, and the retry keeps it.
    slot_1 = [r for r in _ledger_rows(results_dir) if "candidate 1" in r["context"]]
    assert [(r["outcome"], r["context"]) for r in slot_1] == [
        ("rejected", "experiment1 round 0 candidate 1 refine incumbent model_a"),
        (
            "admitted",
            "experiment1 round 0 candidate 1 refine incumbent model_a retry 1",
        ),
    ]
    assert slot_1[0]["detail"] == "no candidate.py written"
    assert slot_1[1]["name"] == "second_idea"


def test_retry_keeps_the_slot_lens(tmp_path, monkeypatch):
    """The retry works the same exploration lens as the attempt it repeats —
    it is the same slot, not a new one in the rotation. (Slot 0 is the
    exploratory slot of a three-slot round; slots 1 and 2 are refinement
    slots, whose retries keep their role — see test_refinement_slots_loop.)"""
    _patch_scoring(monkeypatch)
    briefs = {}

    def fake_spawn(candidate_dir, docs, **kwargs):
        briefs[candidate_dir.name] = docs["brief"]
        if candidate_dir.name != "candidate_0":
            _write_candidate(candidate_dir, name=f"idea_{candidate_dir.name}")
        return True

    _run(tmp_path, monkeypatch, fake_spawn, candidate_count=3)

    assert briefs["candidate_0_retry_1"] == briefs["candidate_0"]
    assert briefs["candidate_0"] != briefs["candidate_1"]


def test_empty_slot_that_stays_empty_is_final(tmp_path, monkeypatch):
    """A slot that writes nothing on its retry either gets no third attempt;
    the other slot's success keeps the round from the all-empty round retry."""
    _patch_scoring(monkeypatch)
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="only_idea")
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=2)

    assert spawns == ["candidate_0", "candidate_1", "candidate_1_retry_1"]
    slot_1 = [r for r in _ledger_rows(results_dir) if "candidate 1" in r["context"]]
    assert [(r["outcome"], r["detail"], r["context"]) for r in slot_1] == [
        (
            "rejected",
            "no candidate.py written",
            "experiment1 round 0 candidate 1 refine incumbent model_a",
        ),
        (
            "rejected",
            "no candidate.py written",
            "experiment1 round 0 candidate 1 refine incumbent model_a retry 1",
        ),
    ]
    assert _manifest_names(results_dir / "models") == [
        "model_a",
        "model_b",
        "only_idea",
    ]


def test_failed_agent_process_is_retried_and_recorded(tmp_path, monkeypatch):
    """An agent process that fails (timeout, non-zero exit) counts as an empty
    attempt: it is recorded in the ledger and the slot is retried once."""
    _patch_scoring(monkeypatch)
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="only_idea")
            return True
        if candidate_dir.name == "candidate_1_retry_1":
            _write_candidate(candidate_dir, name="late_idea")
            return True
        return False

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=2)

    assert spawns == ["candidate_0", "candidate_1", "candidate_1_retry_1"]
    slot_1 = [r for r in _ledger_rows(results_dir) if "candidate 1" in r["context"]]
    assert [(r["outcome"], r["context"]) for r in slot_1] == [
        ("rejected", "experiment1 round 0 candidate 1 refine incumbent model_a"),
        (
            "admitted",
            "experiment1 round 0 candidate 1 refine incumbent model_a retry 1",
        ),
    ]
    assert "agent process failed" in slot_1[0]["detail"]
    assert "late_idea" in _manifest_names(results_dir / "models")


def test_a_timed_out_agents_candidate_is_admitted_through_the_usual_gates(
    tmp_path, monkeypatch
):
    """An agent killed at its time limit may already have written a complete
    candidate (Opus often had, while still checking it). What it left is judged
    by the admission gates like any other file; the slot is not re-spawned."""
    _patch_scoring(monkeypatch)
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        _write_candidate(candidate_dir, name=f"idea_{candidate_dir.name}")
        return candidate_dir.name == "candidate_0"  # candidate_1 "times out"

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=2)

    assert spawns == ["candidate_0", "candidate_1"]
    assert "idea_candidate_1" in _manifest_names(results_dir / "models")


def test_all_slots_empty_after_retries_still_triggers_the_round_retry(
    tmp_path, monkeypatch
):
    """The outer guard survives: when every slot is empty even after its own
    retry, the whole round is re-run in ``iter_0_retry_1``."""
    _patch_scoring(monkeypatch)
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.parent.name + "/" + candidate_dir.name)
        if candidate_dir.parent.name == "iter_0_retry_1":
            _write_candidate(candidate_dir, name=f"idea_{candidate_dir.name}")
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=2)

    assert spawns == [
        "iter_0/candidate_0",
        "iter_0/candidate_1",
        "iter_0/candidate_0_retry_1",
        "iter_0/candidate_1_retry_1",
        "iter_0_retry_1/candidate_0",
        "iter_0_retry_1/candidate_1",
    ]
    assert _manifest_names(results_dir / "models") == [
        "model_a",
        "model_b",
        "idea_candidate_0",
        "idea_candidate_1",
    ]


# ── One repair after a rejection, with the reason in the prompt ───────


def test_rejected_candidate_is_repaired_once_with_the_reason_in_its_prompt(
    tmp_path, monkeypatch
):
    """The first attempt is a near-duplicate of ``model_a``; the repair attempt
    starts from the rejected files, is told the reason verbatim, and its
    (novel) candidate is admitted. Both attempts are in the ledger."""
    _patch_scoring(monkeypatch, near_duplicate_names={"dup_idea"})
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append((candidate_dir, docs))
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="dup_idea", hypothesis="Same as A.")
        elif candidate_dir.name == "candidate_0_repair_1":
            # The rejected attempt's files are already there as a starting point.
            assert (candidate_dir / "candidate.py").read_text(
                encoding="utf-8"
            ) == "# dup_idea\n"
            assert (candidate_dir / "hypothesis.md").read_text(
                encoding="utf-8"
            ) == "Same as A.\n"
            assert (candidate_dir / "model_name.txt").read_text(
                encoding="utf-8"
            ) == "dup_idea\n"
            _write_candidate(candidate_dir, name="fresh_idea", hypothesis="Different.")
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=1)

    assert [d.name for d, _ in spawns] == ["candidate_0", "candidate_0_repair_1"]
    repair_dir, repair_docs = spawns[1]
    assert repair_dir == results_dir / "iter_0" / "candidate_0_repair_1"
    note = repair_docs["attempt_note"]
    assert "predicts like existing model 'model_a'" in note
    assert "p_left RMSE 0.00100 < 0.002" in note
    assert str(results_dir / "iter_0" / "candidate_0") in note
    assert (repair_dir / "ATTEMPT_NOTE.md").read_text(encoding="utf-8") == note
    # The repair is not told not to re-propose the very model it is repairing.
    assert "dup_idea" not in repair_docs["attempted"]

    assert _manifest_names(results_dir / "models") == [
        "model_a",
        "model_b",
        "fresh_idea",
    ]
    rows = _ledger_rows(results_dir)
    assert [(r["name"], r["outcome"], r["context"]) for r in rows] == [
        ("dup_idea", "rejected", "experiment1 round 0 candidate 0 lens 0"),
        ("fresh_idea", "admitted", "experiment1 round 0 candidate 0 lens 0 repair 1"),
    ]
    assert rows[0]["detail"] in note


def test_second_rejection_is_final(tmp_path, monkeypatch):
    """A repair that is rejected again gets no third attempt."""
    _patch_scoring(monkeypatch, near_duplicate_names={"dup_idea", "dup_again"})
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        name = "dup_idea" if candidate_dir.name == "candidate_0" else "dup_again"
        _write_candidate(candidate_dir, name=name)
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=1)

    assert spawns == ["candidate_0", "candidate_0_repair_1"]
    assert _manifest_names(results_dir / "models") == ["model_a", "model_b"]
    rows = _ledger_rows(results_dir)
    assert [(r["name"], r["outcome"], r["context"]) for r in rows] == [
        ("dup_idea", "rejected", "experiment1 round 0 candidate 0 lens 0"),
        ("dup_again", "rejected", "experiment1 round 0 candidate 0 lens 0 repair 1"),
    ]


def test_retry_then_rejection_gets_one_repair(tmp_path, monkeypatch):
    """An empty first attempt, a rejected retry, then a repair: three attempts
    at most, each in the ledger under its own context."""
    _patch_scoring(monkeypatch, near_duplicate_names={"dup_idea"})
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        if candidate_dir.name == "candidate_0_retry_1":
            _write_candidate(candidate_dir, name="dup_idea")
        elif candidate_dir.name == "candidate_0_repair_1":
            _write_candidate(candidate_dir, name="fresh_idea")
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=1)

    assert spawns == ["candidate_0", "candidate_0_retry_1", "candidate_0_repair_1"]
    rows = _ledger_rows(results_dir)
    assert [(r["name"], r["outcome"], r["context"]) for r in rows] == [
        ("iter0_candidate0", "rejected", "experiment1 round 0 candidate 0 lens 0"),
        ("dup_idea", "rejected", "experiment1 round 0 candidate 0 lens 0 retry 1"),
        ("fresh_idea", "admitted", "experiment1 round 0 candidate 0 lens 0 repair 1"),
    ]
    assert "fresh_idea" in _manifest_names(results_dir / "models")


def test_repair_that_writes_nothing_is_final(tmp_path, monkeypatch):
    """A repair attempt that writes no candidate.py is not retried."""
    _patch_scoring(monkeypatch, near_duplicate_names={"dup_idea"})
    spawns = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawns.append(candidate_dir.name)
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="dup_idea")
        else:
            # The copied-in files are the rejected attempt; the agent leaves them.
            # Make that explicit by removing candidate.py as a no-write agent would
            # not — an untouched copy must not be re-admitted either.
            (candidate_dir / "candidate.py").unlink()
        return True

    results_dir = _run(tmp_path, monkeypatch, fake_spawn, candidate_count=1)

    assert spawns == ["candidate_0", "candidate_0_repair_1"]
    rows = _ledger_rows(results_dir)
    assert [(r["outcome"], r["detail"], r["context"]) for r in rows] == [
        (
            "rejected",
            rows[0]["detail"],
            "experiment1 round 0 candidate 0 lens 0",
        ),
        (
            "rejected",
            "no candidate.py written",
            "experiment1 round 0 candidate 0 lens 0 repair 1",
        ),
    ]
    assert "near-duplicate" in rows[0]["detail"]


# ── Units: the verdict, the notes, the prompt ─────────────────────────


def test_admission_verdict_carries_the_rejection_reason(tmp_path, monkeypatch):
    from src.pipelines.inner_loop.model_zoo import (
        Admission,
        _admit_candidate,
        _admit_candidate_with_reason,
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text("models: []\n", encoding="utf-8")
    missing = tmp_path / "candidate_0" / "candidate.py"

    verdict = _admit_candidate_with_reason(
        missing, models_dir, "iter0_candidate0", tmp_path / "responses.csv"
    )
    assert verdict == Admission(admitted=False, reason="no candidate.py written")
    # The boolean form is unchanged for callers that only need the verdict.
    assert (
        _admit_candidate(missing, models_dir, "iter0_candidate0", tmp_path / "r.csv")
        is False
    )


def test_admission_verdict_rejects_inconsistent_states():
    import pytest

    from src.pipelines.inner_loop.model_zoo import Admission

    with pytest.raises(ValueError, match="no rejection reason"):
        Admission(admitted=True, reason="but why")
    with pytest.raises(ValueError, match="needs a rejection reason"):
        Admission(admitted=False, reason="")


def test_repair_note_requires_a_reason(tmp_path):
    import pytest

    from src.pipelines.inner_loop.candidate_agent import _repair_note

    with pytest.raises(ValueError, match="rejection reason"):
        _repair_note(tmp_path, "   ")
    note = _repair_note(tmp_path, "non-finite ELPD-LOO (nan)")
    assert "non-finite ELPD-LOO (nan)" in note
    assert str(tmp_path) in note


def test_prompt_places_the_attempt_note_before_the_context_documents(tmp_path):
    from src.pipelines.inner_loop.candidate_agent import _build_candidate_prompt

    docs = {
        "context": "ctx",
        "brief": "brief",
        "existing_hypotheses": "hyps",
        "critiques": None,
        "attempt_note": "NOTE: this is a repair attempt. Reason: predicts like X.",
    }
    prompt = _build_candidate_prompt(tmp_path / "candidate_0_repair_1", docs)
    note_at = prompt.index(docs["attempt_note"])
    assert prompt.index("Your working directory for this candidate") < note_at
    assert note_at < prompt.index("## CONTEXT.md")

    docs["attempt_note"] = None
    assert "NOTE:" not in _build_candidate_prompt(tmp_path / "candidate_0", docs)
