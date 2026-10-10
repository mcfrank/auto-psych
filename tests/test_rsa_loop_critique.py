"""The RSA loop's critique step (CriticAL, as main's): frames, replicates, the agent round."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.rsa.dataset import load_forced_choice
from src.rsa.fit import mean_probs, posterior_flat
from src.rsa.loop import brief as briefs
from src.rsa.loop.critique import (
    CRITIQUE_COLUMNS,
    ReplicateFrames,
    critique_frame,
    run_critique,
    simulate_choices,
)
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


class StubFit:
    """A posterior without sampling: what `posterior_flat` reads."""

    def __init__(self, n_draws=100, seed=0):
        rng = np.random.default_rng(seed)
        self.param_names = ["alpha", "lapse"]
        self.idata = type("Idata", (), {})()
        self.idata.posterior = {"alpha": rng.uniform(1.5, 2.5, (1, n_draws)),
                                "lapse": rng.uniform(0.02, 0.08, (1, n_draws))}


@pytest.fixture(scope="module")
def twins():
    return load_forced_choice(experiments=["E9_twins"])


@pytest.fixture(scope="module")
def model():
    return RSAModel(SEEDS / "rsa_l1.py")


def test_the_frame_has_design_columns_and_the_chosen_class(twins):
    frame = critique_frame(twins.frame, twins.contexts, twins.choices)
    assert set(frame.columns) <= set(CRITIQUE_COLUMNS) and "choice" in frame.columns
    # Outcome columns a replicate would not redraw are left out.
    assert not {"response", "referent", "display_order"} & set(frame.columns)
    # The second twin's clicks are recorded as the first twin's, as the fit sees them.
    clicked_second = [i for i, c in enumerate(twins.choices) if c == 2]
    assert clicked_second and set(frame["choice"].iloc[clicked_second]) == {1}
    assert len(frame) == len(twins.contexts)


def test_replicates_follow_the_models_class_probabilities(twins, model):
    fit = StubFit()
    sims = simulate_choices(model, fit, twins.contexts, n_replicates=60, seed=1)
    assert sims.shape == (60, len(twins.contexts))
    assert not (sims == 2).any()  # never the second twin: a class is its first object
    expected = np.mean(mean_probs(model, posterior_flat(fit), twins.contexts, 100, by_class=True), axis=0)
    observed = np.bincount(sims.ravel(), minlength=3) / sims.size
    assert np.allclose(observed, expected, atol=0.02)


def test_replicate_frames_are_built_one_at_a_time_and_differ_only_in_choice(twins):
    frame = critique_frame(twins.frame, twins.contexts, twins.choices)
    sims = np.zeros((3, len(frame)), dtype=np.int16)
    reps = ReplicateFrames(frame, sims)
    assert len(reps) == 3
    it = iter(reps)
    first = next(it)
    assert (first["choice"] == 0).all() and first.drop(columns="choice").equals(frame.drop(columns="choice"))
    assert sum(1 for _ in it) == 2


SHARE_FIRST = '''# name: share_first_object
# description: Share of trials choosing object 0 (higher than the model's mean: the model under-produces it).
def test_statistic(df):
    return float((df["choice"] == 0).mean())
'''
N_TRIALS = '''# name: n_trials
# description: The number of trials (the same in every dataset: never a discrepancy).
def test_statistic(df):
    return float(len(df))
'''
BROKEN = '''# name: broken
# description: Raises.
def test_statistic(df):
    return df["no_such_column"].mean()
'''


def test_a_critique_round_retries_once_sets_aside_broken_statistics_and_scores_the_rest(tmp_path, twins, model):
    """The first attempt writes only a broken statistic; the retry is told why and
    writes two. Every observed choice is object 0, which the model cannot produce."""
    calls = []

    def agent(cdir: Path, prompt: str) -> bool:
        calls.append((cdir.name, prompt))
        stats = cdir / "test_stats"
        stats.mkdir(parents=True, exist_ok=True)
        if cdir.name == "critique":
            (stats / "broken.py").write_text(BROKEN)
        else:
            (stats / "share_first_object.py").write_text(SHARE_FIRST)
            (stats / "n_trials.py").write_text(N_TRIALS)
        return True

    out = run_critique(tmp_path / "round_1", spawn=agent, incumbent="rsa_l1", model=model, fitted=StubFit(),
                       hypothesis="Depth-1 RSA.", incumbent_file=SEEDS / "rsa_l1.py", frame=twins.frame,
                       contexts=twins.contexts, choices=[0] * len(twins.contexts),
                       responses_path=tmp_path / "responses.csv", n_proposals=2, n_replicates=50)
    assert [c[0] for c in calls] == ["critique", "critique_retry_1"]
    assert "set aside" in calls[1][1] and "no_such_column" in calls[1][1]
    assert "## CRITIQUE_CONTEXT.md" in calls[0][1] and "rsa_l1" in calls[0][1]
    s = out.status
    assert (s["status"], s["attempts"], s["n_statistics"], s["n_evaluated"]) == ("critiqued", 2, 2, 2)
    assert s["n_significant"] == 1
    assert "share_first_object" in out.critiques_md and "Significant discrepancies" in out.critiques_md
    result = json.loads((tmp_path / "round_1" / "critique_retry_1" / "ppc_results.json").read_text())
    top = result["results"][0]
    assert top["name"] == "share_first_object" and top["t_observed"] == 1.0 and top["significant"]
    assert (tmp_path / "round_1" / "critique" / "broken_statistics" / "broken.py").exists()


def test_a_critique_agent_that_writes_nothing_leaves_the_round_without_a_critique(tmp_path, twins, model):
    out = run_critique(tmp_path / "round_1", spawn=lambda d, p: True, incumbent="rsa_l1", model=model,
                       fitted=StubFit(), hypothesis="", incumbent_file=SEEDS / "rsa_l1.py", frame=twins.frame,
                       contexts=twins.contexts, choices=twins.choices, responses_path=tmp_path / "r.csv",
                       n_replicates=20)
    assert out.critiques_md is None
    assert out.status["status"] == "no_critique" and out.status["attempts"] == 2
    assert "no usable test statistic" in out.status["reason"]


def test_the_candidates_get_the_critique_with_mains_instruction(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    live = [briefs.ZooModel("rsa_l1", "Depth-1 RSA.", tmp_path / "rsa_l1.py", "best")]
    kw = dict(role="explore", lens="a lens", context="# Context\n", live=live, pruned=[], incumbent="rsa_l1",
              ledger=ledger)
    with_critique = briefs.write_docs(tmp_path / "a", critiques="# Critique of `rsa_l1`\n", **kw)
    prompt = briefs.build_prompt(tmp_path / "a", with_critique)
    assert "## critiques.md\n\n# Critique of `rsa_l1`" in prompt
    assert "prioritise a hypothesis that addresses one of" in with_critique["brief"]
    assert "posterior-predictive critique" in with_critique["context"]
    assert (tmp_path / "a" / "critiques.md").exists()
    without = briefs.write_docs(tmp_path / "b", **kw)
    assert "critiques.md" not in briefs.build_prompt(tmp_path / "b", without)
    assert not (tmp_path / "b" / "critiques.md").exists()


def test_the_settings_are_mains():
    from src.pipelines.inner_loop import critique_round as main
    from src.rsa.loop.orchestrator import LoopConfig

    cfg = LoopConfig(responses_path=Path("r"), seed_models_dir=Path("s"), results_dir=Path("o"))
    assert (cfg.n_critique_proposals, cfg.critique_alpha, cfg.n_critique_replicates) == (
        main.CRITIQUE_N_PROPOSALS, main.CRITIQUE_SIGNIFICANCE_ALPHA, main.CRITIQUE_PPC_REPLICATES)
