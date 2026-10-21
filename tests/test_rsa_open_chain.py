"""A chain end to end (src/rsa_open/chain.py): design, simulated collection and the inner loop over two
experiments, with a fake designer and theorist, scripted pilot participants, a small EIG and a canned
link check (each has its own tests); every fit and simulation is real."""

import json
import math

import pytest
import yaml

from src.rsa.fit import FitSettings
from tests.test_rsa_open_design import SEEDS
from tests.test_rsa_open_designer import FakeDesigner, spec_for
from tests.test_rsa_open_loop_orchestrator import ported
from tests.test_rsa_open_pilot import Scripted

SETTINGS = FitSettings(num_warmup=150, num_samples=150, num_chains=2, seed=0, dense_mass=True)


def specs():
    """The designer's two paradigms: forced choice, then production (each in its pilot's time)."""
    choice = spec_for()
    production = spec_for()
    production["response"] = {"type": "production", "link": "production",
                              "prompt": "Which word would you use to point out the marked creature?"}
    return choice, production


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    import src.rsa_open.design as design
    import src.rsa_open.pilot as pilot
    import src.rsa_open.tools as tools
    from src.rsa_open.chain import Chain, ChainConfig, GroundTruth
    from src.rsa_open.corpus import Block
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from tests.test_rsa_open_corpus import simulated_block

    mp = pytest.MonkeyPatch()
    mp.setattr(design, "DEFAULT_DRAWS", 16)
    mp.setattr(design, "DEFAULT_SCENARIOS", 100)
    mp.setattr(design, "reference_report", lambda ctx, **kw: {"bits_per_dollar": 1e-9})
    mp.setattr(tools, "link_check_report", lambda ws, spec, **kw: {
        "paradigm_sha256": __import__("src.rsa_open.paradigm", fromlist=["x"]).load_paradigm(spec).sha256,
        "link": "x", "theory": "x", "n_trials": 1, "converged": True, "recovered": True, "misses": {}})
    real_pilot = pilot.run_pilot
    mp.setattr(pilot, "run_pilot", lambda paradigm, lists, out, *, n_participants, participant_factory=None,
               max_workers=6, timing=None: real_pilot(paradigm, lists, out, n_participants=n_participants,
                                                      participant_factory=lambda i: Scripted(lists, i, paradigm=paradigm),
                                                      max_workers=1, timing=timing))

    root = tmp_path_factory.mktemp("chain")
    seeds = root / "seeds"
    seeds.mkdir()
    entries = []
    for name, rationale in (("literal_listener", "Listeners pick among the objects the word is true of."),
                            ("rsa_l1", "Pragmatic listeners reason about an informative speaker.")):
        (seeds / f"{name}.py").write_text(ported(name))
        entries.append(dict(name=name, rationale=rationale))
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}))
    gt_path = root / "ground_truth.py"
    gt_path.write_text(ported("rsa_l1_salience"))
    truth = {"alpha": 2.0, "w_features": 0.5, "w_familiar": 0.0}
    gt = TheoryModel(gt_path)
    literature = simulated_block(gt, "E", likelihood("choice"), 120, 1, {"temperature": 1.0, "lapse": 0.05})
    designer = FakeDesigner(list(specs()))
    theorist_calls = []

    def theorist(cdir, prompt):  # proposes nothing: the chain's own machinery is under test
        theorist_calls.append(cdir)
        return True

    cfg = ChainConfig(chain="dry", seeds_dir=seeds, literature=literature, out_dir=root / "chain",
                      ground_truth=GroundTruth(gt_path, truth), n_experiments=2, settings=SETTINGS,
                      pilot_participants=3, loop=dict(max_iterations=1, candidate_count=1, fit_time_limit_sec=None))
    c = Chain(cfg, designer_spawn=designer, theorist_spawn=theorist)
    exports = c.run()
    yield c, exports, designer, theorist_calls
    mp.undo()


@pytest.mark.slow  # two experiments: designs, simulated data, every theory fitted to every fold
def test_two_experiments_run_end_to_end_and_carry_their_theories(chain):
    c, exports, designer, theorist_calls = chain
    assert designer.sessions == 2 and len(exports) == 2
    for n, link in ((1, "choice"), (2, "production")):
        exp = c.exp(n)
        record = json.loads((exp / "design" / "frozen" / "record.json").read_text())
        assert record["chain"] == "dry" and record["experiment"] == n and record["review_required"] == (n == 1)
        from src.rsa_open.loop.fitting import read_corpus

        (block,) = read_corpus(exp / "data" / "block.json")
        assert block.name == f"exp{n}" and block.link.name == link and block.n == 120 * 6
        names = [b.name for b in read_corpus(exp / "corpus.json")]
        assert names == ["E"] + [f"exp{k}" for k in range(1, n + 1)]
    # Experiment 2 starts from experiment 1's live set and continues its ledger.
    live1 = set(exports[0]["live"])
    assert {e["name"] for e in yaml.safe_load((c.exp(2) / "models_input" / "models_manifest.yaml").read_text())["models"]} == live1
    ledger2 = (c.exp(2) / "loop" / "attempted_hypotheses.jsonl").read_text()
    assert ledger2.startswith((c.exp(1) / "loop" / "attempted_hypotheses.jsonl").read_text())
    # The second design saw the first paradigm and each theory's standing on it.
    brief2 = (c.exp(2) / "design" / "cycle_1" / "logs" / "brief_1.md").read_text()
    assert "1_paradigm.json" in brief2
    fits_md = (c.exp(2) / "design" / "cycle_1" / "inputs" / "history" / "1_fits.md").read_text()
    assert all(name in fits_md for name in live1)
    standing = json.loads((c.exp(2) / "loop" / "history.json").read_text())[-1]["standing"]
    assert all(set(s["by_block"]) == {"E", "exp1", "exp2"} for s in standing.values() if s["by_block"])


@pytest.mark.slow
def test_a_finished_chain_resumes_without_redoing_anything(chain):
    c, exports, designer, _ = chain
    sessions = designer.sessions
    assert c.run() == exports and designer.sessions == sessions
