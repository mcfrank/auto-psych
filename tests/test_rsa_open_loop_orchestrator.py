"""The open inner loop end to end (src/rsa_open/loop/orchestrator.py), with a fake theorist."""

import json

import pytest
import yaml

from src.rsa.fit import FitSettings
from tests.test_rsa_open_design import SEEDS

SETTINGS = FitSettings(num_warmup=150, num_samples=150, num_chains=2, seed=0, dense_mass=True)


def ported(name):
    from src.rsa_open.port import port

    return port((SEEDS / f"{name}.py").read_text(), name=f"{name}.py")


class FakeTheorist:
    """Slot 1 proposes rsa_l1_salience (novel); slot 2 copies rsa_l1's hypothesis (refused, then a
    repair that copies its file: not novel); slot 3 writes nothing, twice."""

    def __init__(self):
        self.calls = []

    def __call__(self, cdir, prompt):
        self.calls.append((cdir.name, prompt))
        cdir.mkdir(parents=True, exist_ok=True)
        if cdir.name == "candidate_1":
            (cdir / "candidate.py").write_text(ported("rsa_l1_salience"))
            (cdir / "hypothesis.md").write_text("Listeners expect speakers to talk about salient, feature-rich objects.")
            (cdir / "model_name.txt").write_text("salient_prior_listener\n")
        elif cdir.name == "candidate_2":
            (cdir / "candidate.py").write_text(ported("rsa_l1"))
            (cdir / "hypothesis.md").write_text("Pragmatic listeners reason about an informative speaker.")
        elif cdir.name == "candidate_2_repair_1":
            (cdir / "hypothesis.md").write_text("A pragmatic listener with a sharper speaker, as its own mechanism.")
        return True


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.fitting import write_corpus
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig
    from tests.test_rsa_open_corpus import simulated_block

    root = tmp_path_factory.mktemp("open_loop")
    seeds = root / "seeds"
    seeds.mkdir()
    entries = []
    for name, rationale in (("literal_listener", "Listeners pick among the objects the word is true of."),
                            ("rsa_l1", "Pragmatic listeners reason about an informative speaker.")):
        (seeds / f"{name}.py").write_text(ported(name))
        entries.append(dict(name=name, rationale=rationale))
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}))
    gt_path = root / "gt.py"
    gt_path.write_text(ported("rsa_l1_salience"))
    gt = TheoryModel(gt_path)
    blocks = [simulated_block(gt, "E", likelihood("choice"), 120, 1, {"temperature": 1.0, "lapse": 0.05}),
              simulated_block(gt, "exp1", likelihood("production"), 120, 2, {"temperature": 1.5, "lapse": 0.05})]
    corpus = write_corpus(root / "corpus.json", blocks)
    theorist = FakeTheorist()
    cfg = OpenLoopConfig(corpus_path=corpus, seed_models_dir=seeds, results_dir=root / "loop", max_iterations=1,
                         candidate_count=3, settings=SETTINGS, fit_time_limit_sec=None,
                         paradigm_notes={"exp1": "People chose the word to say for a marked object."})
    loop = OpenLoop(cfg, theorist)
    final = loop.run()
    return loop, final, theorist


@pytest.mark.slow  # every theory is fitted to the corpus and to each of 5 CV folds
def test_a_round_admits_refuses_repairs_retries_and_exports(run):
    loop, final, theorist = run
    ledger = [json.loads(line) for line in (loop.dir / "attempted_hypotheses.jsonl").read_text().splitlines()]
    outcomes = {(e["name"], e["outcome"]) for e in ledger}
    assert ("salient_prior_listener", "admitted") in outcomes
    rejected = [e for e in ledger if e["outcome"] == "rejected"]
    assert any("word for word" in e["detail"] for e in rejected)  # slot 2 copied rsa_l1's hypothesis
    assert any("not novel" in e["detail"] for e in rejected)  # its repair, a copy of rsa_l1's file
    assert any(e["detail"] == "the agent wrote no candidate.py" for e in rejected)
    assert [c for c, _ in theorist.calls].count("candidate_3") == 1
    assert "candidate_3_retry_1" in [c for c, _ in theorist.calls]
    export = json.loads((loop.dir / "export.json").read_text())
    assert export["best_model"] == final["best_model"] and (loop.dir / "best_model.py").exists()
    assert set(export["live"]) <= {"literal_listener", "rsa_l1", "salient_prior_listener"}
    s = final["standing"]
    assert all({"sel_elpd", "eligible", "by_block", "trusted"} <= set(d) for d in s.values())
    assert set(next(iter(s.values()))["by_block"]) == {"E", "exp1"}
    assert [h["round"] for h in json.loads((loop.dir / "history.json").read_text())] == [-1, 0, 1]


@pytest.mark.slow
def test_the_theorist_s_prompt_describes_every_paradigm_and_the_theory_api(run):
    loop, _, theorist = run
    prompt = dict(theorist.calls)["candidate_1"]
    for text in ("def listener(params, ctx)", "### E:", "### exp1:", "choose the word to say", "production",
                 "-m src.rsa_open.loop.check_candidate", "within 4 standard errors",
                 "People chose the word to say for a marked object."):
        assert text in prompt, text
    assert "def choice_probs(params, ctx):\n    ..." not in prompt
    assert (loop.dir / "data" / "exp1.jsonl").exists()
