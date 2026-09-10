"""The inner loop keeps a ledger of every hypothesis it tried, and the next
round's candidates read it.

Pruned models used to vanish from ``existing_hypotheses.md`` the moment they
left the zoo, so the next round (and the next experiment) re-proposed them: in
the iteration-2 recovery sweep 11 of the 13 re-proposed names in the weakest
cell had already been pruned in that cell. The ledger
(``model_loop/attempted_hypotheses.jsonl``) records one line per candidate
slot and per pruning, starts from the ledger the previous experiment carried,
and is rendered into every candidate brief as "already tried — do not
re-propose". MCMC, agents and scoring are stubbed; only the bookkeeping runs.
"""

from __future__ import annotations

import json

import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import write_responses, write_seed_models

INHERITED = {
    "name": "old_idea",
    "outcome": "pruned",
    "detail": "50.0 nats behind model_a (10.0× dse)",
    "hypothesis": "People dislike long runs.",
    "context": "experiment1 round 1",
}


def _manifest_names(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return [e["name"] for e in data["models"]]


def _patch_scoring(monkeypatch):
    """Stateless scoring stubs keyed on the current manifest.

    ``model_a`` always wins. Any model named ``idea_one`` or ``carried_c`` sits
    30 nats behind at dse 5 (pruned at 2·dse); everything else is within noise.
    """

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        return {
            "posteriors": {n: (1.0 if n == "model_a" else 0.0) for n in names},
            "elpd_loo": {n: (-10.0 if n == "model_a" else -40.0) for n in names},
            "n_trials": 2,
        }

    def fake_compare(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        rows = {}
        rank = 0
        for n in ["model_a"] + [m for m in names if m != "model_a"]:
            if n not in names:
                continue
            far = n in ("idea_one", "carried_c")
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

    monkeypatch.setattr(pymc_orchestrator, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(pymc_orchestrator, "compare_table", fake_compare)
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(pymc_orchestrator, "evict_fit_cache", lambda name: None)
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda name, models_dir: object()
    )


def _patch_candidates(monkeypatch, names_by_round, briefs):
    """Round r's single agent proposes ``names_by_round[r]``; briefs are captured."""

    def fake_spawn(candidate_dir, docs, **kwargs):
        round_idx = int(candidate_dir.parent.name.split("_")[1])
        name = names_by_round[round_idx]
        briefs.append(docs)
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            f"People use heuristic {name}. It is a single mechanism.\n",
            encoding="utf-8",
        )
        (candidate_dir / "model_name.txt").write_text(name + "\n", encoding="utf-8")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)


def _ledger_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ledger_inherits_records_every_slot_and_reaches_the_next_brief(
    tmp_path, monkeypatch
):
    seed_dir = write_seed_models(tmp_path)
    (seed_dir / LEDGER_FILENAME).write_text(
        json.dumps(INHERITED) + "\n", encoding="utf-8"
    )
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)
    briefs = []
    _patch_candidates(monkeypatch, {0: "idea_one", 1: "idea_two"}, briefs)
    # Round 1's candidate predicts like model_a → rejected as a near-duplicate.
    monkeypatch.setattr(
        pymc_orchestrator,
        "_min_prediction_rmse",
        lambda name, *a, **k: ("model_a", 0.001) if name == "idea_two" else (None, float("inf")),
    )
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=2,
        candidate_count=1,
        enable_critique=False,
        ledger_context="experiment2",
    )

    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    assert [(r["name"], r["outcome"]) for r in rows] == [
        ("old_idea", "pruned"),
        ("idea_one", "admitted"),
        ("idea_one", "pruned"),
        ("idea_two", "rejected"),
    ]
    assert rows[0] == INHERITED
    assert "30.0 nats behind model_a" in rows[2]["detail"]
    assert "near-duplicate of model_a" in rows[3]["detail"]
    assert rows[3]["hypothesis"].startswith("People use heuristic idea_two")
    assert rows[1]["context"] == "experiment2 round 0 candidate 0 lens 0"
    assert rows[2]["context"] == "experiment2 round 0"

    # Round 0's brief carries the inherited entry; round 1's also carries the
    # model pruned in round 0, with its margin — under the do-not-re-propose
    # heading — and the text is injected into the agent's prompt.
    assert "old_idea" in briefs[0]["attempted"]
    assert "idea_one" not in briefs[0]["attempted"]
    assert "idea_one" in briefs[1]["attempted"]
    assert "30.0 nats behind model_a" in briefs[1]["attempted"]
    assert "do not re-propose" in briefs[1]["attempted"].lower()
    round1_file = results_dir / "iter_1" / "candidate_0" / "attempted_hypotheses.md"
    assert round1_file.read_text(encoding="utf-8") == briefs[1]["attempted"]
    prompt = pymc_orchestrator._build_candidate_prompt(
        results_dir / "iter_1" / "candidate_0", briefs[1]
    )
    assert "## attempted_hypotheses.md" in prompt
    assert "idea_one" in prompt
    # The pruned model is still readable for audit.
    assert (results_dir / "models" / "pruned" / "idea_one.py").exists()


def test_carried_models_outside_protected_names_can_be_pruned(tmp_path, monkeypatch):
    seed_dir = write_seed_models(tmp_path, names=("model_a", "model_b", "carried_c"))
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=0,
        enable_critique=False,
        protected_names={"model_a", "model_b"},
    )

    assert _manifest_names(results_dir / "models") == ["model_a", "model_b"]
    assert (results_dir / "models" / "pruned" / "carried_c.py").exists()
    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    assert [(r["name"], r["outcome"]) for r in rows] == [("carried_c", "pruned")]


def test_every_seeded_model_is_protected_when_no_protected_names_are_given(
    tmp_path, monkeypatch
):
    seed_dir = write_seed_models(tmp_path, names=("model_a", "model_b", "carried_c"))
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=0,
        enable_critique=False,
    )

    assert _manifest_names(results_dir / "models") == ["model_a", "model_b", "carried_c"]
    assert _ledger_rows(results_dir / LEDGER_FILENAME) == []
