"""A round has refinement slots (depth) as well as exploratory ones (breadth).

In the September 2026 sweep the exported best model never changed from its
starting seed (0 of 27 scoring steps): the novelty gate, the ledger's "do not
re-propose" and pruning all push toward *new* mechanisms, and nothing let a
partially correct one be improved. Each round now allocates its candidate
slots by role (``model_zoo.slot_roles``): exploratory slots walk the lens
battery as before; two slots refine the incumbent, named in the brief; one
slot refines a non-incumbent model of the agent's choosing, from a menu of the
live and pruned models with their full hypotheses. Below four slots the
refinement slots are given up one at a time — never the exploratory one.

The agent says in ``hypothesis.md`` which model it refined; nothing in the
pipeline parses that or branches on it. What differs per slot is the prompt
and the ledger context (the slot's *assignment*, not the agent's output).
MCMC, agents and scoring are stubbed; only the round bookkeeping runs.
"""

from __future__ import annotations

import json

import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.candidate_agent import DEFAULT_CANDIDATE_HINTS
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import write_responses


def _manifest_names(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return [e["name"] for e in data["models"]]


def _write_seed_models(tmp_path):
    """Two seeds with stated hypotheses (the menu renders them in full)."""
    seed_dir = tmp_path / "seed_models"
    seed_dir.mkdir()
    entries = [
        {"name": "model_a", "rationale": "People count heads; A wins on this data."},
        {"name": "model_b", "rationale": "People count alternations; B is the rival."},
    ]
    for entry in entries:
        (seed_dir / f"{entry['name']}.py").write_text(
            f"# stub {entry['name']}\n", encoding="utf-8"
        )
    (seed_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": entries}, sort_keys=False), encoding="utf-8"
    )
    return seed_dir


def _patch_scoring(monkeypatch, *, losers=()):
    """``model_a`` is always the incumbent. A model named in ``losers`` sits
    30 nats behind at dse 5 and is pruned; everything else is within noise."""

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        return {
            "posteriors": {n: (1.0 if n == "model_a" else 0.0) for n in names},
            "elpd_loo": {n: (-10.0 if n == "model_a" else -11.0) for n in names},
            "n_trials": 2,
        }

    def fake_compare(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        rows = {}
        rank = 0
        for n in ["model_a"] + [m for m in names if m != "model_a"]:
            if n not in names:
                continue
            far = n in losers
            rows[n] = {
                "rank": rank,
                "elpd_loo": -10.0 - (30.0 if far else 1.0 * rank),
                "elpd_diff": 0.0 if n == "model_a" else (30.0 if far else 1.0),
                "dse": 0.0 if n == "model_a" else 5.0,
                "weight": 1.0 if n == "model_a" else 0.0,
                "loo_unreliable": False,
            }
            rank += 1
        return rows

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, models_dir: object())
    monkeypatch.setattr(
        model_zoo, "_min_prediction_rmse", lambda *a, **k: (None, float("inf"))
    )


LONG_HYPOTHESIS = " ".join(
    f"Sentence {i}: the run-length mechanism also predicts a consequence "
    f"for sequences whose structure is unusual in way number {i}."
    for i in range(20)
)


def _write_candidate(candidate_dir, *, name, hypothesis):
    (candidate_dir / "candidate.py").write_text(f"# {name}\n", encoding="utf-8")
    (candidate_dir / "hypothesis.md").write_text(hypothesis + "\n", encoding="utf-8")
    (candidate_dir / "model_name.txt").write_text(name + "\n", encoding="utf-8")


def _ledger_rows(results_dir):
    path = results_dir / LEDGER_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_three_slot_round_is_one_exploratory_one_incumbent_one_chosen(
    tmp_path, monkeypatch
):
    """Two rounds of three slots. Slot 0 explores (lens 0, then lens 1: the
    walk covers exploratory slots only); slot 1 refines the incumbent
    ``model_a``, named in its brief; slot 2 chooses from a menu that lists
    the live rival ``model_b`` and — in round 1 — the model round 0 admitted
    and pruned, with its full hypothesis, its margin and its pruned source."""
    assert len(LONG_HYPOTHESIS) > 2000
    _patch_scoring(monkeypatch, losers={"runs_idea"})
    seed_dir = _write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    docs_by_slot = {}

    def fake_spawn(candidate_dir, docs, **kwargs):
        rnd = int(candidate_dir.parent.name.split("_")[1])
        idx = int(candidate_dir.name.split("_")[1])
        docs_by_slot[(rnd, idx)] = docs
        if (rnd, idx) == (0, 0):
            _write_candidate(candidate_dir, name="runs_idea", hypothesis=LONG_HYPOTHESIS)
        else:
            _write_candidate(
                candidate_dir,
                name=f"idea_{rnd}_{idx}",
                hypothesis=f"Refining something, round {rnd} slot {idx}.",
            )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"
    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=2,
        candidate_count=3,
        enable_critique=False,
        ledger_context="experiment1",
    )
    models_dir = results_dir / "models"

    # ── Slot 0: exploratory, and the lens walk is over exploratory slots only.
    for rnd, lens in ((0, 0), (1, 1)):
        brief = docs_by_slot[(rnd, 0)]["brief"]
        assert DEFAULT_CANDIDATE_HINTS[lens] in brief
        assert "blended mega-model is not a hypothesis" in brief
        assert docs_by_slot[(rnd, 0)]["menu"] is None
        # An exploratory slot carries the ledger's "tried before" section.
        assert docs_by_slot[(rnd, 0)]["attempted"].startswith("# Tried before")

    # ── Slot 1: refine the incumbent, named explicitly, with its hypothesis.
    for rnd in (0, 1):
        docs = docs_by_slot[(rnd, 1)]
        brief = docs["brief"]
        assert "Refine the incumbent" in brief
        assert "`model_a`" in brief
        assert "People count heads; A wins on this data." in brief
        assert str(models_dir / "model_a.py") in brief
        assert "blended mega-model is not a hypothesis" not in brief
        assert not any(hint in brief for hint in DEFAULT_CANDIDATE_HINTS)
        assert docs["attempted"] is None
        assert docs["menu"] is not None

    # ── Slot 2: refine a model of the agent's choosing, from the menu.
    round0 = docs_by_slot[(0, 2)]
    assert "of your choosing" in round0["brief"]
    assert docs_by_slot[(0, 2)]["attempted"] is None
    menu0 = round0["menu"]
    assert "model_b" in menu0
    assert "People count alternations; B is the rival." in menu0
    assert str(models_dir / "model_b.py") in menu0
    assert "model_a" not in menu0.split("## Live", 1)[1].split("## Pruned", 1)[0]
    assert "do not re-propose" not in menu0.lower()
    assert "runs_idea" not in menu0  # not yet proposed
    # Round 1's menu lists the model round 0 pruned — in full, with its
    # margin and its source under models/pruned/.
    menu1 = docs_by_slot[(1, 2)]["menu"]
    assert "runs_idea" in menu1
    assert LONG_HYPOTHESIS in menu1
    assert "30.0 nats behind model_a" in menu1
    assert str(models_dir / "pruned" / "runs_idea.py") in menu1
    assert (models_dir / "pruned" / "runs_idea.py").exists()
    assert "…" not in menu1
    # The same file is on disk for audit, and the prompt inlines it.
    menu_file = results_dir / "iter_1" / "candidate_2" / "refinement_menu.md"
    assert menu_file.read_text(encoding="utf-8") == menu1
    assert not (results_dir / "iter_1" / "candidate_2" / "attempted_hypotheses.md").exists()
    assert (results_dir / "iter_1" / "candidate_0" / "attempted_hypotheses.md").exists()
    assert not (results_dir / "iter_1" / "candidate_0" / "refinement_menu.md").exists()

    # ── The ledger records the slot's assignment (not the agent's choice).
    contexts = {
        r["name"]: r["context"] for r in _ledger_rows(results_dir) if r["outcome"] == "admitted"
    }
    assert contexts["runs_idea"] == "experiment1 round 0 candidate 0 lens 0"
    assert contexts["idea_0_1"] == "experiment1 round 0 candidate 1 refine incumbent model_a"
    assert contexts["idea_0_2"] == "experiment1 round 0 candidate 2 refine chosen"
    assert contexts["idea_1_0"] == "experiment1 round 1 candidate 0 lens 1"
    assert contexts["idea_1_1"] == "experiment1 round 1 candidate 1 refine incumbent model_a"
    assert contexts["idea_1_2"] == "experiment1 round 1 candidate 2 refine chosen"


def test_six_slot_round_has_three_exploratory_lenses_that_never_repeat(
    tmp_path, monkeypatch
):
    """At six slots (the P45 setting): slots 0-2 exploratory with distinct
    lenses, slots 3-4 refine the incumbent, slot 5 is agent-chosen."""
    _patch_scoring(monkeypatch)
    seed_dir = _write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    briefs = {}

    def fake_spawn(candidate_dir, docs, **kwargs):
        idx = int(candidate_dir.name.split("_")[1])
        briefs[idx] = docs["brief"]
        _write_candidate(candidate_dir, name=f"idea_{idx}", hypothesis="One claim.")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"
    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=6,
        enable_critique=False,
        ledger_context="experiment1",
    )

    lenses_fired = []
    for idx in (0, 1, 2):
        fired = [h for h in DEFAULT_CANDIDATE_HINTS if h in briefs[idx]]
        assert len(fired) == 1, idx
        lenses_fired.append(fired[0])
    assert len(set(lenses_fired)) == 3
    for idx in (3, 4):
        assert "Refine the incumbent" in briefs[idx] and "`model_a`" in briefs[idx]
    assert "of your choosing" in briefs[5]
    contexts = [r["context"] for r in _ledger_rows(results_dir) if r["outcome"] == "admitted"]
    assert contexts == [
        "experiment1 round 0 candidate 0 lens 0",
        "experiment1 round 0 candidate 1 lens 1",
        "experiment1 round 0 candidate 2 lens 2",
        "experiment1 round 0 candidate 3 refine incumbent model_a",
        "experiment1 round 0 candidate 4 refine incumbent model_a",
        "experiment1 round 0 candidate 5 refine chosen",
    ]


def test_refinement_slot_retry_and_repair_keep_their_role(tmp_path, monkeypatch):
    """Slot 1 (incumbent refinement) writes nothing, then its retry is a
    near-duplicate of the incumbent, then its repair is admitted: every
    attempt keeps the slot's role in its brief and its ledger context."""
    _patch_scoring(monkeypatch)
    monkeypatch.setattr(
        model_zoo,
        "_min_prediction_rmse",
        lambda name, *a, **k: ("model_a", 0.0001) if name == "same_as_a" else (None, float("inf")),
    )
    seed_dir = _write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    briefs = {}

    def fake_spawn(candidate_dir, docs, **kwargs):
        briefs[candidate_dir.name] = docs["brief"]
        if candidate_dir.name == "candidate_0":
            _write_candidate(candidate_dir, name="explore_idea", hypothesis="New.")
        elif candidate_dir.name == "candidate_1_retry_1":
            _write_candidate(candidate_dir, name="same_as_a", hypothesis="A again.")
        elif candidate_dir.name == "candidate_1_repair_1":
            _write_candidate(candidate_dir, name="a_refined", hypothesis="A, refined.")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"
    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=2,
        enable_critique=False,
        ledger_context="experiment1",
    )

    for attempt in ("candidate_1", "candidate_1_retry_1", "candidate_1_repair_1"):
        assert "Refine the incumbent" in briefs[attempt], attempt
    slot_1 = [r for r in _ledger_rows(results_dir) if "candidate 1" in r["context"]]
    assert [(r["name"], r["outcome"], r["context"]) for r in slot_1] == [
        ("iter0_candidate1", "rejected", "experiment1 round 0 candidate 1 refine incumbent model_a"),
        ("same_as_a", "rejected", "experiment1 round 0 candidate 1 refine incumbent model_a retry 1"),
        ("a_refined", "admitted", "experiment1 round 0 candidate 1 refine incumbent model_a repair 1"),
    ]
    assert _manifest_names(results_dir / "models") == [
        "model_a",
        "model_b",
        "explore_idea",
        "a_refined",
    ]
