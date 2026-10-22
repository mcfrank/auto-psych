"""A theory fitted to many paradigms (src/rsa_open/corpus.py): shared theory parameters, each
block's own link parameters, grouped CV within blocks, and the accountability rule."""

import arviz as az
import jax
import numpy as np
import pytest

from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import prior_draws
from tests.test_rsa_open_design import SEEDS


@pytest.fixture(scope="module")
def theory(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    path = tmp_path_factory.mktemp("theory") / "rsa_l1_salience.py"
    path.write_text(port((SEEDS / "rsa_l1_salience.py").read_text(), name="rsa_l1_salience.py"))
    return TheoryModel(path)


TRUTH = {"alpha": 2.0, "w_features": 0.5, "w_familiar": 0.0}


def simulated_block(theory, name, link, n, seed, link_params, units_per=4):
    from src.rsa_open.corpus import Block
    from src.rsa_open.fit import simulate

    pool = plain_pool()
    rng = np.random.default_rng(seed)
    scenes = [pool[i] for i in rng.choice(len(pool), n)]
    responses = simulate(theory, link, scenes, {**TRUTH, **link_params}, jax.random.PRNGKey(seed))
    return Block(name, link, tuple(scenes), tuple(responses), tuple(f"u{i // units_per}" for i in range(n)))


@pytest.fixture(scope="module")
def blocks(theory):
    from src.rsa_open.links import likelihood

    choice, rating = likelihood("choice"), likelihood("rating", scale_points=5)
    return [simulated_block(theory, "E", choice, 400, 1, {"temperature": 1.0, "lapse": 0.05}),
            simulated_block(theory, "exp1", rating, 400, 2,
                            {"temperature": 2.5, "lapse": 0.05, "center": 0.0, "spread": 0.8})]


def test_the_joint_likelihood_is_each_block_s_under_its_own_link_parameters(theory, blocks):
    from numpyro.infer.util import log_density

    from src.rsa_open.corpus import _corpus_model
    from src.rsa_open.fit import trial_log_lik

    params = {**prior_draws(theory.params, 1, 3)[0]}
    links = {}
    for b in blocks:
        draw = prior_draws(b.link.params, 1, 4 + len(links))[0]
        links[b.name] = draw
        params.update({b.param(k): v for k, v in draw.items()})
    joint, _ = log_density(_corpus_model, (theory, blocks), {}, params)
    priors = sum(float(d.log_prob(params[k])) for k, d in theory.params.items()) + sum(
        float(d.log_prob(params[b.param(k)])) for b in blocks for k, d in b.link.params.items())
    per_block = sum(float(trial_log_lik(theory, b.link, b.data, {**{k: params[k] for k in theory.params},
                                                                  **links[b.name]}).sum()) for b in blocks)
    assert float(joint) - priors == pytest.approx(per_block, rel=1e-4)


def test_folds_take_whole_units_and_balance_each_block(blocks):
    from src.rsa_open.corpus import assign_folds

    for b in blocks:
        fold = assign_folds(b, k=5, seed=0)
        units = np.asarray(b.units)
        for u in set(b.units):
            assert len(set(fold[units == u])) == 1
        sizes = np.bincount(fold, minlength=5)
        assert sizes.min() > 0 and sizes.max() - sizes.min() <= 4
    with pytest.raises(ValueError, match="fewer than"):
        assign_folds(blocks[0].subset(range(8)), k=5)


def test_held_out_scores_are_the_log_posterior_mean_probability(theory, blocks):
    from src.rsa_open.corpus import CorpusFit, block_draws, heldout_lpd
    from src.rsa_open.fit import pointwise_log_lik

    b = blocks[1].subset(range(30))
    draws = {k: np.asarray([d[k] for d in prior_draws(theory.params, 6, 1)]) for k in theory.params}
    draws.update({b.param(k): np.asarray([d[k] for d in prior_draws(b.link.params, 6, 2)]) for k in b.link.params})
    idata = az.from_dict(posterior={k: v[None] for k, v in draws.items()})
    fitted = CorpusFit("t", [b.name], idata, list(draws), [])
    ll = pointwise_log_lik(theory, b.link, b.data, block_draws(fitted.flat(), theory, b))
    expected = np.log(np.exp(ll).mean(axis=0))
    np.testing.assert_allclose(heldout_lpd(theory, fitted, b), expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="no link parameters"):
        heldout_lpd(theory, fitted, blocks[0])


def test_a_theory_much_worse_on_an_earlier_paradigm_is_not_eligible(blocks):
    from src.rsa_open.corpus import CorpusCV, standing

    rng = np.random.default_rng(0)
    base = {b.name: rng.normal(-1.0, 0.3, b.n) for b in blocks}
    results = {
        "steady": CorpusCV({k: v.copy() for k, v in base.items()}, True),
        # best on the new paradigm, and far behind on E (every E trial 0.5 nats worse)
        "specialist": CorpusCV({"E": base["E"] - 0.5, "exp1": base["exp1"] + 0.2}, True),
        # a little worse on average, by a noisy amount (a constant gap has a clustered SE of 0)
        "close": CorpusCV({k: v + rng.normal(-0.002, 0.05, v.shape) for k, v in base.items()}, True),
    }
    s = standing(results, blocks)
    assert s["specialist"]["sel_diff"] == 0.0  # it leads on the newest block
    assert not s["specialist"]["eligible"] and s["specialist"]["guard_failures"] == ["E"]
    assert s["steady"]["eligible"] and s["close"]["eligible"]
    assert s["steady"]["by_block"]["E"]["behind"] == 0.0


def test_blocks_refuse_inconsistent_trials_and_names(blocks):
    from src.rsa_open.corpus import Block, check_blocks

    b = blocks[0]
    with pytest.raises(ValueError, match="responses"):
        Block("x", b.link, b.contexts, b.responses[:-1], b.units)
    with pytest.raises(ValueError, match="'@'"):
        Block("a@b", b.link, b.contexts, b.responses, b.units)
    with pytest.raises(ValueError, match="repeat"):
        check_blocks([b, b])


def test_a_paradigm_s_collected_trials_become_a_block():
    from src.rsa_open.corpus import paradigm_block
    from src.rsa_open.screens import trial_lists
    from tests.test_rsa_open_screens import answer, paradigm_of, units_of

    p = paradigm_of("rating")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=10)
    rng = np.random.default_rng(0)
    rows = [{"phase": t["phase"], "unit": t["unit"], "objects": t["objects"], "heard": t["heard"],
             "response": answer(t, rng)} for lst in lists["lists"] for t in lst["trials"]]
    block = paradigm_block("exp1", p, rows)
    assert block.n == 60 and block.link.name == "rating" and set(block.units) <= {f"unit{u}" for u in range(10)}
    block.data  # the responses parse as the link's data


@pytest.mark.slow  # NUTS on two simulated blocks
def test_a_joint_fit_recovers_the_shared_theory_and_each_block_s_link(theory, blocks):
    from src.rsa.fit import FitSettings
    from src.rsa_open.corpus import fit_corpus

    fitted = fit_corpus(theory, blocks, FitSettings(num_warmup=400, num_samples=400, num_chains=2, seed=0,
                                                     dense_mass=True))
    assert fitted.converged, fitted.convergence_problems
    flat = fitted.flat()
    lo, hi = {k: np.quantile(v, 0.01) for k, v in flat.items()}, {k: np.quantile(v, 0.99) for k, v in flat.items()}
    for name, true in (("alpha", 2.0), ("w_features", 0.5), ("temperature@E", 1.0), ("temperature@exp1", 2.5)):
        assert lo[name] <= true <= hi[name], (name, lo[name], hi[name])
    assert hi["temperature@E"] < lo["temperature@exp1"] or np.mean(flat["temperature@exp1"]) > np.mean(flat["temperature@E"])
    assert fitted.pointwise("exp1").shape[1] == 400


def test_the_literature_is_the_reference_block_its_temperature_fixed_at_1(theory, blocks, tmp_path):
    """PI 2026-10-22: on E the link's temperature is fixed at 1 (its lapse stays free); every new
    paradigm's temperature is free, relative to E. On E alone a free temperature trades off against
    a theory's own sharpness (run A: rsa_l1_shared_prior's fit stalled; alpha 0.15 against 1.4)."""
    from numpyro import handlers

    from src.rsa_open.corpus import Block, _corpus_model, block_draws, literature_block
    from src.rsa_open.loop.fitting import read_corpus, write_corpus

    E = literature_block()
    assert E.fixed_params == {"temperature": 1.0} and set(E.free_link_params) == {"lapse"}
    E_sim = Block("E", blocks[0].link, blocks[0].contexts, blocks[0].responses, blocks[0].units,
                  fixed=(("temperature", 1.0),))
    corpus = [E_sim, blocks[1]]
    trace = handlers.trace(handlers.seed(_corpus_model, 0)).get_trace(theory, corpus)
    sampled = {k for k, v in trace.items() if v["type"] == "sample" and not v.get("is_observed")}
    assert "temperature@E" not in sampled and {"lapse@E", "temperature@exp1", "lapse@exp1"} <= sampled
    flat = {"alpha": np.ones(3), "w_features": np.zeros(3), "w_familiar": np.zeros(3), "lapse@E": np.full(3, 0.1)}
    assert block_draws(flat, theory, E_sim)["temperature"].tolist() == [1.0, 1.0, 1.0]
    # The corpus file keeps it (and a block without fixed parameters is written as before).
    (back_E, back_1) = read_corpus(write_corpus(tmp_path / "c.json", corpus))
    assert back_E.fixed_params == {"temperature": 1.0} and back_1.fixed_params == {}
    assert '"fixed"' not in write_corpus(tmp_path / "one.json", [blocks[1]]).read_text()
    assert E_sim.subset([0, 1]).fixed_params == {"temperature": 1.0}
    with pytest.raises(ValueError, match="not a parameter of the choice link"):
        Block("E", blocks[0].link, blocks[0].contexts, blocks[0].responses, blocks[0].units, fixed=(("kappa", 1.0),))
