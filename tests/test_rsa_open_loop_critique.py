"""The open loop's critic (src/rsa_open/loop/critique.py): the newest paradigm's rows, replicates
through its link, the closed critic's rules."""

import json

import numpy as np
import pytest

from tests.test_rsa_open_corpus import TRUTH, simulated_block


class FakeFit:
    """A CorpusFit's posterior draws, without a fit."""

    def __init__(self, flat):
        self._flat = flat

    def flat(self):
        return self._flat


@pytest.fixture(scope="module")
def setting(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from tests.test_rsa_open_loop_orchestrator import ported

    root = tmp_path_factory.mktemp("critique")
    (root / "rsa_l1_salience.py").write_text(ported("rsa_l1_salience"))
    theory = TheoryModel(root / "rsa_l1_salience.py")
    block = simulated_block(theory, "exp1", likelihood("production"), 160, 3, {"temperature": 1.5, "lapse": 0.05})
    rng = np.random.default_rng(0)
    flat = {k: np.full(40, float(v)) + 0.01 * rng.normal(size=40) for k, v in TRUTH.items() if k in theory.params}
    flat.update({"temperature@exp1": np.full(40, 1.5), "lapse@exp1": np.full(40, 0.05)})
    return root, theory, block, FakeFit(flat)


def test_the_frame_keeps_the_design_and_names_the_response(setting):
    from src.rsa_open.loop.critique import OUTCOME, critique_frame

    _, _, block, _ = setting
    df = critique_frame(block)
    assert len(df) == block.n and {"objects", "heard", "unit", "target", "utterance"} <= set(df.columns)
    assert OUTCOME["production"] == "utterance"
    assert json.loads(df["objects"][0]) == [list(r) for r in block.contexts[0].objects]


def test_replicates_redraw_only_the_response_through_the_block_s_link(setting):
    from src.rsa_open.loop.critique import simulate_outcomes

    _, theory, block, fit = setting
    reps = simulate_outcomes(theory, fit, block, n_replicates=12, seed=0)
    assert len(reps) == 12 and all(len(r) == block.n for r in reps)
    observed = [dict(r)["utterance"] for r in block.responses]
    assert any(r != observed for r in reps)
    # Every simulated word is a real message of its scene (never the sink).
    for r in reps[:3]:
        for ctx, u in zip(block.contexts, r):
            assert 0 <= u < ctx.shape[1] - (1 if ctx.needs_sink else 0)


def _stat(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(f"# name: {name}\n# description: {name}.\ndef test_statistic(df):\n{body}\n")


def test_a_round_s_critique_scores_the_critic_s_statistics(setting, tmp_path):
    from src.rsa_open.loop.critique import run_critique

    root, theory, block, fit = setting
    prompts = []

    def critic(cdir, prompt):
        prompts.append(prompt)
        _stat(cdir / "test_stats", "mean_word", "    return float(df['utterance'].mean())")
        _stat(cdir / "test_stats", "heard_named", "    return float((df['utterance'] == df['heard']).mean())")
        return True

    out = run_critique(tmp_path / "round_1", spawn=critic, incumbent="rsa_l1_salience", theory=theory, fitted=fit,
                       hypothesis="Salient referents.", incumbent_file=root / "rsa_l1_salience.py", block=block,
                       paradigm_note="People chose a word for a marked creature.", data_path=tmp_path / "exp1.jsonl",
                       n_replicates=40, seed=1)
    assert out.status["status"] == "critiqued" and out.status["n_evaluated"] == 2 and out.status["block"] == "exp1"
    assert out.critiques_md and (tmp_path / "round_1" / "critique" / "critiques.md").exists()
    result = json.loads((tmp_path / "round_1" / "critique" / "ppc_results.json").read_text())
    assert result["n_replicates"] == 40
    prompt = prompts[0]
    for text in ("one theory must predict all of them", "`exp1`, the newest one", "`production`",
                 "`utterance`: the index of the feature whose word was chosen", "People chose a word"):
        assert text.lower() in prompt.lower(), text
    assert "clicks the object they" not in prompt  # not the closed critic's click-only framing


def test_a_critic_that_writes_nothing_leaves_the_round_without_a_critique(setting, tmp_path):
    from src.rsa_open.loop.critique import CRITIQUE_STATUS_NONE, run_critique

    root, theory, block, fit = setting
    calls = []
    out = run_critique(tmp_path, spawn=lambda cdir, prompt: calls.append(cdir.name) or True, incumbent="x",
                       theory=theory, fitted=fit, hypothesis="", incumbent_file=root / "rsa_l1_salience.py",
                       block=block, paradigm_note="", data_path=tmp_path / "exp1.jsonl", n_replicates=10)
    assert out.critiques_md is None and out.status["status"] == CRITIQUE_STATUS_NONE and calls == ["critique", "critique_retry_1"]
