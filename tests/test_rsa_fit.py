"""Fitting memo models with numpyro: recovery, trial order, zero-probability choices."""

import numpy as np
import pytest

from src.rsa.context import Context
from src.rsa.fit import FitSettings, ZeroProbabilityChoice, compare, fit, prepare
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"

SIMPLE = dict(objects=((0, 0), (0, 1), (1, 1)), feature_names=("hat", "glasses"))
COMPLEX = dict(
    objects=((0, 0, 1), (0, 1, 1), (1, 1, 0)), feature_names=("hat", "glasses", "mustache")
)
SIZE = dict(objects=((1, 0), (1, 1), (0, 1), (0, 1)), feature_names=("a", "b"))


def design():
    return [
        Context(**SIMPLE, utterance=1),
        Context(**SIMPLE, utterance=0),
        Context(**COMPLEX, utterance=1),
        Context(**COMPLEX, utterance=2),
        Context(**SIZE, utterance=0),
        Context(**SIZE, utterance=1),
    ]


def simulate(model, params, contexts, n_per_context, seed=0):
    rng = np.random.default_rng(seed)
    from src.rsa.context import group_by_shape

    out_ctx, out_choice = [], []
    for group in group_by_shape(contexts).values():
        p = np.asarray(model.group_probs(params, group), dtype=float)
        for row, idx in enumerate(group.indices):
            probs = p[row] / p[row].sum()
            for c in rng.choice(len(probs), size=n_per_context, p=probs):
                out_ctx.append(contexts[idx])
                out_choice.append(int(c))
    return out_ctx, out_choice


def test_prepare_reduces_trials_to_displays_and_patterns():
    size, simple = Context(**SIZE, utterance=0), Context(**SIMPLE, utterance=1)
    # A relabelled copy of a display is the same display to every model.
    relabelled = Context(objects=SIMPLE["objects"], feature_names=("x", "y"), utterance=1, item="other")
    prep = prepare([size, simple, simple, relabelled, simple], [3, 1, 1, 1, 2])
    assert prep.width == 4
    assert [g.shape for g in prep.groups] == [(3, 3), (4, 2)]
    assert [len(g.indices) for g in prep.groups] == [1, 1]  # two distinct displays
    # Objects 2 and 3 of SIZE are identical: choosing 3 is observed as class 2.
    patterns = list(zip(prep.pattern_row[prep.trial_pattern], prep.pattern_class[prep.trial_pattern]))
    assert patterns == [(1, 2), (0, 1), (0, 1), (0, 1), (0, 2)]
    assert sorted(prep.pattern_count.tolist()) == [1, 1, 3]


def test_a_choice_among_identical_objects_is_observed_as_its_class():
    """The data cannot say which of two identical twins was chosen, and no
    model can tell them apart: the likelihood of a twin choice is the twins'
    summed probability, whichever copy the row names."""
    from src.rsa.fit import class_probs

    twins = Context(
        objects=((0, 1, 1), (1, 0, 1), (1, 0, 1)),
        feature_names=("hat", "glasses", "mustache"),
        utterance=0,
    )
    prep = prepare([twins, twins], [1, 2])
    assert prep.pattern_class.tolist() == [1] and prep.pattern_count.tolist() == [2]
    p = np.array([[0.1, 0.45, 0.45], [0.1, 0.45, 0.45]])
    np.testing.assert_allclose(class_probs(p, np.stack([prep.groups[0].classes[0]] * 2)),
                               [[0.1, 0.9, 0.0]] * 2)


def test_a_choice_outside_the_context_raises():
    with pytest.raises(ValueError, match="not an object index"):
        prepare([Context(**SIMPLE, utterance=1)], [3])


def test_a_model_with_zero_probability_for_an_observed_choice_is_refused(tmp_path):
    body = (SEEDS / "literal_listener.py").read_text().replace(
        'with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])',
        'jnp.where(ctx.is_prior > 0, uniform, heard) + 0.0 * params["lapse"]',
    )
    path = tmp_path / "strict_literal.py"
    path.write_text(body)
    # The plain face (object 0) is not wearing glasses: impossible for a
    # literal listener without lapses.
    with pytest.raises(ZeroProbabilityChoice, match="probability 0"):
        fit(RSAModel(path), [Context(**SIMPLE, utterance=1)], [0], FitSettings(num_warmup=10, num_samples=10))


@pytest.mark.slow
def test_nuts_recovers_alpha_and_loo_prefers_the_generating_depth():
    truth = RSAModel(SEEDS / "rsa_l2.py")
    contexts, choices = simulate(truth, {"alpha": 3.0, "lapse": 0.05}, design(), 150)
    settings = FitSettings(num_warmup=500, num_samples=500, num_chains=2, seed=3)
    fits = {
        name: fit(RSAModel(SEEDS / f"{name}.py"), contexts, choices, settings)
        for name in ("rsa_l2", "rsa_l1", "literal_listener")
    }
    l2 = fits["rsa_l2"]
    assert l2.converged, l2.convergence_problems
    alpha = np.asarray(l2.idata.posterior["alpha"]).ravel()
    assert np.quantile(alpha, 0.025) < 3.0 < np.quantile(alpha, 0.975)
    # Log-likelihood is stored per (display, choice) pattern; LOO is per trial.
    assert l2.idata.constant_data["trial_pattern"].shape == (len(choices),)
    assert l2.loo().loo["loo_i"].shape == (len(choices),)
    table = compare(fits)
    assert table.index[0] == "rsa_l2"
    assert table.index[-1] == "literal_listener"


@pytest.mark.slow
def test_pattern_loo_equals_trial_level_loo():
    """Expanding pattern-level PSIS-LOO to trials gives what PSIS-LOO over the
    per-trial log-likelihood gives: identical columns, identical terms."""
    import arviz as az

    from src.models.loo_reliability import loo_diagnostics

    truth = RSAModel(SEEDS / "rsa_l1.py")
    contexts, choices = simulate(truth, {"alpha": 2.0, "lapse": 0.1}, design(), 40, seed=4)
    fitted = fit(truth, contexts, choices, FitSettings(num_warmup=200, num_samples=200, num_chains=2))
    tp = fitted.idata.constant_data["trial_pattern"].values
    ll = np.asarray(fitted.idata.log_likelihood["choice"])[..., tp]
    per_trial = az.from_dict(posterior={"alpha": np.asarray(fitted.idata.posterior["alpha"])},
                             log_likelihood={"choice": ll})
    want, got = loo_diagnostics(per_trial), fitted.loo()
    # Up to float32 rounding (the draws are stored in float32).
    np.testing.assert_allclose(got.elpd_loo, want.elpd_loo, rtol=1e-4)
    np.testing.assert_allclose(got.loo["se"], want.loo["se"], rtol=1e-4)
    np.testing.assert_allclose(got.loo["p_loo"], want.loo["p_loo"], rtol=1e-3)
    np.testing.assert_allclose(got.loo["loo_i"].values, want.loo["loo_i"].values, rtol=1e-4, atol=1e-5)
    assert (got.n_points, got.n_bad_k, got.n_exact) == (want.n_points, want.n_bad_k, want.n_exact)
    assert len(fitted.idata.constant_data["pattern_count"]) < len(choices) / 5
