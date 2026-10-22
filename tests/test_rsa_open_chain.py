"""A chain end to end (src/rsa_open/chain.py): design, simulated collection and the inner loop over two
experiments, with a fake designer and theorist, scripted pilot participants, a small EIG and a canned
link check (each has its own tests); every fit and simulation is real."""

import json
import math

import numpy as np
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
                      pilot_participants=3, design_settings=SETTINGS, loop=dict(max_iterations=1, candidate_count=1, fit_time_limit_sec=None))
    c = Chain(cfg, designer_spawn=designer, theorist_spawn=lambda loop_dir: theorist)
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
    # Experiment 2's design reads experiment 1's loop fits: same corpus file, so the carried theories are hits.
    assert (c.exp(2) / "prior_corpus.json").read_bytes() == (c.exp(1) / "loop" / "corpus.json").read_bytes()
    record = json.loads((c.exp(2) / "design_screened_out.json").read_text())
    assert record["screened_out"] == [] and set(record["fits"].values()) == {"the previous loop's fit"}
    assert not (c.exp(2) / ".fit_cache").exists()  # nothing refitted for the design
    # Experiment 2's design scored the bar too: a seed the loop pruned is still in its context.
    context2 = json.loads((c.exp(2) / "design" / "cycle_1" / "context" / "context.json").read_text())
    assert set(context2["theories"]) == {"literal_listener", "rsa_l1"}
    if live1 != {"literal_listener", "rsa_l1"}:
        assert "kept in every design as the bar" in brief2
    standing = json.loads((c.exp(2) / "loop" / "history.json").read_text())[-1]["standing"]
    assert all(set(s["by_block"]) == {"E", "exp1", "exp2"} for s in standing.values() if s["by_block"])


@pytest.mark.slow
def test_a_finished_chain_resumes_without_redoing_anything(chain):
    c, exports, designer, _ = chain
    sessions = designer.sessions
    assert c.run() == exports and designer.sessions == sessions


def test_an_unfinished_stage_is_set_aside_with_its_fit_cache_carried(tmp_path):
    from src.rsa_open.chain import set_aside

    assert set_aside(tmp_path / "loop") is None
    for attempt in (1, 2):
        (tmp_path / "loop" / ".fit_cache").mkdir(parents=True, exist_ok=True)
        (tmp_path / "loop" / ".fit_cache" / "a.nc").write_text("fit")
        (tmp_path / "loop" / "round_0").mkdir()
        aside = set_aside(tmp_path / "loop", keep=(".fit_cache",))
        assert aside == tmp_path / f"loop.abandoned_{attempt}" and (aside / "round_0").is_dir()
        assert (tmp_path / "loop" / ".fit_cache" / "a.nc").read_text() == "fit"
        assert not (tmp_path / "loop" / "round_0").exists()


def test_the_chain_command_starts_from_its_chain_s_seeds_and_designs_against_the_bar(tmp_path, monkeypatch):
    import src.rsa_open.run_chain as run_chain
    from src.rsa_open.chain import GroundTruth
    from src.rsa_open.seeds import OUT

    monkeypatch.setattr(run_chain, "load_ground_truth", lambda d: GroundTruth(tmp_path / "gt.py", {"alpha": 1.0}))
    monkeypatch.setattr(run_chain, "literature_block", lambda path: ("E from", path))
    args = run_chain.Args(chain=1, run_dir=tmp_path / "run", private_dir=tmp_path / "private",
                          agent_root=tmp_path, literature=tmp_path / "all_trials.csv", no_sandbox=True)
    c = run_chain.build(args)
    assert c.cfg.seeds_dir == OUT / "chain_1" and c.cfg.bar_dir == OUT / "bar"
    assert c.cfg.literature == ("E from", tmp_path / "all_trials.csv") and c.cfg.chain == "chain_1"
    assert c.cfg.loop == {"max_iterations": 5, "candidate_count": 6, "cv_folds": 5}
    assert callable(c.theorist_spawn(tmp_path / "run" / "experiment1" / "loop"))


def test_each_design_reads_the_timing_of_every_finished_live_experiment(tmp_path):
    """PLAN.md, "Timing digest": the closed campaign's records plus every finished experiment's, this
    chain's and the other chains' (simulated collection records none: simulated people take no time)."""
    from src.rsa_open.chain import Chain, ChainConfig, GroundTruth
    from src.rsa_open.screens import trial_lists
    from src.rsa_open.timing import CLOSED_CAMPAIGN
    from tests.test_rsa_open_screens import paradigm_of, units_of

    def chain(name, others=()):
        cfg = ChainConfig(chain=name, seeds_dir=tmp_path, literature=None, out_dir=tmp_path / name,
                          ground_truth=GroundTruth(tmp_path / "gt.py", {}), other_chains=tuple(others))
        return Chain(cfg, designer_spawn=None, theorist_spawn=None)

    p = paradigm_of("choice")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=2)
    a, b = chain("chain_0", [tmp_path / "chain_1"]), chain("chain_1", [tmp_path / "chain_0"])
    for c in (a, b):
        frozen = c.exp(1) / "design" / "frozen"
        frozen.mkdir(parents=True)
        (frozen / "lists.json").write_text(json.dumps(lists))
    assert a.timing(1) == list(CLOSED_CAMPAIGN)
    rec = a.record_timing(1, [100.0, 150.0, 200.0])
    assert rec.n_people == 3 and rec.median_s == 150.0 and rec.source == "chain_0, experiment 1"
    assert a.timing(2) == list(CLOSED_CAMPAIGN) + [rec]
    assert b.timing(1) == list(CLOSED_CAMPAIGN) + [rec]  # another chain's finished experiment counts
    with pytest.raises(FileExistsError):
        a.record_timing(1, [1.0])  # once


def test_a_design_leaves_out_a_theory_whose_fit_fails_or_runs_out_of_time_on_record(tmp_path, monkeypatch):
    from src.rsa.loop.fitting import FitTimeLimitExceeded, ModelFailure
    from src.rsa_open.chain import Chain, ChainConfig, GroundTruth
    from tests.test_rsa_open_loop_critique import FakeFit

    cfg = ChainConfig(chain="c", seeds_dir=tmp_path, literature=None, out_dir=tmp_path / "chain",
                      ground_truth=GroundTruth(tmp_path / "gt.py", {}))
    c = Chain(cfg, designer_spawn=None, theorist_spawn=None)
    names = ["literal_listener", "rsa_l1", "slow", "broken"]
    for name in names:
        (tmp_path / f"{name}.py").write_text(ported("rsa_l1" if name in ("slow", "broken") else name))
    monkeypatch.setattr(Chain, "blocks_before", lambda self, n: [])
    monkeypatch.setattr("src.rsa_open.chain.write_corpus", lambda path, blocks: path)
    monkeypatch.setattr(Chain, "design_theories", lambda self, n: {k: (tmp_path / f"{k}.py", "h") for k in names})

    def fit(self, n, path, name, corpus):
        if name == "slow":
            raise FitTimeLimitExceeded("slow: the fit did not finish within 120 minutes")
        if name == "broken":
            raise ModelFailure("ValueError: nan")
        return FakeFit({"alpha": np.ones(50)}), "a design fit"

    monkeypatch.setattr(Chain, "design_fit", fit)
    context = c.design_context(1)
    assert set(context) == {"literal_listener", "rsa_l1"}
    record = json.loads((c.exp(1) / "design_screened_out.json").read_text())
    assert [s["theory"] for s in record["screened_out"]] == ["slow", "broken"]
    assert "120 minutes" in record["screened_out"][0]["reason"]
    names[:] = ["rsa_l1", "slow"]
    with pytest.raises(RuntimeError, match="needs two"):
        c.design_context(1)
