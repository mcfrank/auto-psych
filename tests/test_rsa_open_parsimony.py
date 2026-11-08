"""The one-SE rule (PI 2026-11-08; src/rsa_open/corpus.py `simplest_adequate`): of the trusted,
eligible theories within ``parsimony_k`` clustered SEs of the best on the newest block, the one with
the fewest parameters leads. Pruning still measures distance from the best fit, and never prunes
the incumbent; each scored step records the best by fit alone beside it."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.rsa_open.corpus import CorpusCV

N = 40


def _blocks():
    return [SimpleNamespace(name="E", units=tuple(range(N)), n=N), SimpleNamespace(name="exp1", units=tuple(range(N)), n=N)]


def _pointwise():
    rng = np.random.default_rng(3)
    base = rng.normal(-1.0, 0.3, N)

    def behind(per_trial):  # noise of SD 0.2 per trial (a clustered SE of about 1.3 nats), mean gap fixed
        d = rng.normal(0.0, 0.2, N)
        return base + 0.05 - (d - d.mean()) - per_trial

    return {  # (E, exp1) held-out lpd per trial; every theory is level on E
        "rich": (base.copy(), base + 0.05),       # best on exp1, 4 parameters
        "lean": (base.copy(), behind(0.01)),      # 0.4 nats behind, within one SE, 2 parameters
        "lean_too": (base.copy(), behind(0.02)),  # 0.8 nats behind, within one SE, 2 parameters
        "bare": (base.copy(), base - 0.5),        # 22 nats behind with no noise, 1 parameter
    }


PARAMS = {"rich": 4, "lean": 2, "lean_too": 2, "bare": 1}


def test_the_simplest_theory_within_one_se_of_the_best_leads():
    from src.models.clustered_se import cluster_dse
    from src.rsa_open.corpus import simplest_adequate

    results = {k: CorpusCV({"E": e, "exp1": x}, True) for k, (e, x) in _pointwise().items()}
    pool, blocks = list(results), _blocks()
    units = np.arange(N)
    for n in ("lean", "lean_too"):  # the fixture is what it says: within one SE of the best
        gap = results["rich"].elpd("exp1") - results[n].elpd("exp1")
        assert 0 < gap <= cluster_dse(results["rich"].pointwise["exp1"], results[n].pointwise["exp1"], units)
    assert simplest_adequate(pool, results, blocks, PARAMS) == "lean"  # fewest parameters; bare is too far behind
    assert simplest_adequate(pool, results, blocks, PARAMS, k=0.0) == "rich"  # k = 0: the best fit
    assert simplest_adequate(["rich", "bare"], results, blocks, PARAMS) == "rich"
    with pytest.raises(ValueError):
        simplest_adequate([], results, blocks, PARAMS)


def _loop(tmp_path):
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    loop = OpenLoop.__new__(OpenLoop)
    loop.cfg = OpenLoopConfig(corpus_path=tmp_path, seed_models_dir=tmp_path, results_dir=tmp_path, max_iterations=1,
                              prune_dse_multiplier=2.0)
    loop.dir, loop.models_dir = tmp_path, tmp_path / "models"
    loop.models_dir.mkdir()
    loop.blocks = _blocks()
    loop.cv = {k: CorpusCV({"E": e, "exp1": x}, True) for k, (e, x) in _pointwise().items()}
    loop.cv_problems, loop.admission_limit_sec, loop.chance_cv = {}, None, None
    loop.live_cfg = SimpleNamespace(time_limit_sec=None)
    loop.live = {}
    for name, k in PARAMS.items():
        (loop.models_dir / f"{name}.py").write_text("# theory\n")
        loop.live[name] = SimpleNamespace(name=name, hypothesis=name, source="seed",
                                          theory=SimpleNamespace(params={f"p{i}": 0 for i in range(k)}),
                                          summary=SimpleNamespace(converged=True, timing={}))
    loop.ledger = HypothesisLedger.create(tmp_path / "attempted_hypotheses.jsonl", inherit_from=None)
    loop.history, loop.step = [], 0
    loop._write_manifest = lambda: None
    return loop


def test_the_loop_s_incumbent_follows_the_rule_and_the_fit_alone_best_is_recorded(tmp_path):
    loop = _loop(tmp_path)
    s = loop.standing()
    assert loop.incumbent(s) == "lean" and loop.fit_best(s) == "rich"
    loop.cfg.parsimony_k = 0.0
    assert loop.incumbent(s) == "rich"


def test_an_ineligible_simpler_theory_cannot_lead(tmp_path):
    loop = _loop(tmp_path)
    e, x = loop.cv["lean"].pointwise["E"], loop.cv["lean"].pointwise["exp1"]
    loop.cv["lean"] = CorpusCV({"E": e - 1.0, "exp1": x}, True)  # far behind on E: fails the guard
    s = loop.standing()
    assert not s["lean"]["eligible"] and loop.incumbent(s) == "lean_too"


def test_the_end_prunes_by_distance_from_the_best_fit_and_keeps_the_incumbent(tmp_path):
    loop = _loop(tmp_path)
    entry = loop.end()
    assert entry["best_model"] == "lean" and entry["fit_best"] == "rich"
    assert entry["n_params"] == {n: PARAMS[n] for n in loop.live} and entry["parsimony_k"] == 1.0
    assert "bare" not in loop.live and {"rich", "lean"} <= set(loop.live)  # bare is distinguishable from the best
    assert "vs rich" in (tmp_path / "attempted_hypotheses.jsonl").read_text()  # measured from the best fit


# --------------------------------------------------------------------------- what the theorists are told

def test_the_prompt_states_the_goal_and_the_rule():
    prompt = open("src/rsa_open/loop/prompts/open_theory.md").read()
    assert "## What a good theory is, and how the loop chooses" in prompt
    assert "one-SE rule" in prompt and "fewest parameters" in prompt and "simplifying is progress" in prompt
    assert prompt.index("## What a good theory is") < prompt.index("## Step 1")


def _brief(tmp_path, role):
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
    from src.rsa.loop.brief import ZooModel
    from src.rsa_open.loop import brief

    ledger = HypothesisLedger.create(tmp_path / f"{role}.jsonl", inherit_from=None)
    inc = ZooModel("rsa_l1", "Pragmatic listeners.", tmp_path / "rsa_l1.py", "THE INCUMBENT; 1 parameter; best on exp1")
    other = ZooModel("lit", "Literal listeners.", tmp_path / "lit.py", "0 parameters; 3.0 ± 1.0 nats behind the best")
    return brief.write_docs(tmp_path / role, role=role, lens="a lens" if role == "explore" else None, context="# Context",
                            live=[inc, other], pruned=[], incumbent="rsa_l1", ledger=ledger)["brief"]


def test_a_refinement_may_be_a_simplification(tmp_path):
    from src.rsa.loop.brief import SLOT_EXPLORE, SLOT_REFINE_CHOSEN, SLOT_REFINE_INCUMBENT

    incumbent = _brief(tmp_path, SLOT_REFINE_INCUMBENT)
    assert "a simplification (removing a component" in incumbent and "takes the lead" in incumbent
    assert "why it should fit better" not in incumbent
    assert "may be a simplification" in _brief(tmp_path, SLOT_REFINE_CHOSEN)
    assert "simplification" not in _brief(tmp_path, SLOT_EXPLORE)


def test_the_slot_rewrite_fails_loudly_when_the_closed_wording_changes():
    from src.rsa.loop.brief import SLOT_REFINE_INCUMBENT
    from src.rsa_open.loop.brief import _parsimony

    with pytest.raises(RuntimeError, match="refinement wording changed"):
        _parsimony("# This slot: refine the incumbent\n\nSomething else.\n", SLOT_REFINE_INCUMBENT)


def test_the_theorists_see_each_theory_s_parameters_and_why_the_incumbent_leads(tmp_path):
    loop = _loop(tmp_path)
    s = loop.standing()
    lead = loop.incumbent(s)
    assert loop._describe("lean", s, lead).startswith("THE INCUMBENT (within 1 SE of the best, with fewer parameters); "
                                                       "2 parameters; ")
    assert loop._describe("rich", s, lead).startswith("4 parameters; best on exp1")
    assert loop._describe("bare", s, lead).startswith("1 parameter; ")
