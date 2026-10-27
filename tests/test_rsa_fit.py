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



def test_an_unknown_init_strategy_raises():
    with pytest.raises(ValueError, match="init"):
        FitSettings(init="median")



def _double_well(depth_b, width_b):
    """A 1-d energy with wells at x = -2 (depth 0, unit curvature) and x = +2 (depth ``depth_b``,
    curvature 1 / width_b**2), joined by a high barrier at 0."""
    import jax
    import jax.numpy as jnp

    def energy(x):
        a = 0.5 * (x[0] + 2.0) ** 2
        b = depth_b + 0.5 * ((x[0] - 2.0) / width_b) ** 2
        return -jnp.logaddexp(-a, -b) + 50.0 * jnp.exp(-4.0 * x[0] ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(energy))
    return (lambda x: tuple(np.asarray(v, dtype=np.float64) for v in value_and_grad(jnp.asarray(x)))), \
        (lambda x: np.asarray(jax.hessian(energy)(jnp.asarray(x)), dtype=np.float64))


def test_find_modes_merges_starts_in_one_well_and_weighs_each_mode_by_laplace_mass():
    from src.rsa.fit import find_modes

    energy_and_grad, hessian = _double_well(depth_b=1.0, width_b=2.0)
    starts = [np.array([v]) for v in (-3.0, -1.0, -2.5, 1.0, 3.5)]
    modes, failed = find_modes(energy_and_grad, hessian, starts)
    assert failed == 0
    assert [m.starts for m in modes] == [3, 2]
    assert np.allclose([m.x[0] for m in modes], [-2.0, 2.0], atol=0.1)  # each pulled slightly by the other well
    # Laplace log mass: -energy - 0.5 log det(H); well b is wider (x2) and 1 nat higher.
    assert np.isclose(modes[1].log_mass - modes[0].log_mass, -1.0 + np.log(2.0), atol=0.15)


def test_a_minor_mode_holding_real_mass_is_a_convergence_problem():
    from src.rsa.fit import find_modes, minor_mode_problem

    starts = [np.array([-3.0]), np.array([3.0])]
    heavy, _ = find_modes(*_double_well(depth_b=1.0, width_b=1.0), starts)  # b holds e^-1 / (1 + e^-1) = 27%
    problem = minor_mode_problem(heavy)
    assert problem is not None and "27" in problem
    light, _ = find_modes(*_double_well(depth_b=8.0, width_b=1.0), starts)  # b holds ~0.03%
    assert minor_mode_problem(light) is None
    assert minor_mode_problem(find_modes(*_double_well(1.0, 1.0), [np.array([-3.0])])[0]) is None  # one mode
    # The straight-line test can split one curved ridge in two; a mode the chains reached is no problem.
    assert minor_mode_problem(heavy, visited=[True, True]) is None
    assert minor_mode_problem(heavy, visited=[True, False]) is not None


def test_a_mode_inside_the_posterior_draws_is_visited():
    from src.rsa.fit import visited_by_draws

    rng = np.random.default_rng(0)
    draws = rng.normal([1.0, 5.0], [0.1, 2.0], size=(4000, 2))
    assert visited_by_draws(draws, [np.array([1.1, 8.0]), np.array([2.0, 5.0])]) == [True, False]


@pytest.mark.slow
def test_the_map_start_is_the_best_optimum_found_and_a_map_fit_records_its_modes():
    # init="map" starts every chain at the best of L-BFGS optimisations from the prior median
    # and uniform starts. On the full consolidation superset a third of uniform starts fell into
    # a minor mode ~500 nats of barrier away (R-hat 1.74); with interior off, the median start
    # itself lands in a mode 19.6 nats below the best one.
    import json

    from src.rsa.fit import _numpyro_model, map_start

    model = RSAModel(SEEDS / "rsa_l1.py")
    contexts, choices = simulate(model, {"alpha": 3.0, "lapse": 0.05}, design(), 100)
    prep = prepare(contexts, choices)
    start, modes, failed = map_start(model, prep, seed=0)
    assert set(start) == {"alpha", "lapse"}
    assert 1.5 < start["alpha"] < 6.0
    assert sum(m.starts for m in modes) + failed == 21  # one start steps to alpha ~ 44, where rsa_l1 is NaN

    from numpyro.infer.util import log_density

    def density(params):
        return float(log_density(_numpyro_model, (model, prep), {}, params)[0])

    assert density(start) >= density({"alpha": 3.0, "lapse": 0.05})
    fitted = fit(model, contexts, choices, FitSettings(num_warmup=0, num_samples=1, num_chains=2, init="map"))
    assert fitted.idata.posterior["alpha"].shape == (2, 1)
    recorded = json.loads(fitted.idata.posterior.attrs["map_modes"])
    assert recorded[0]["starts"] == modes[0].starts and "share" in recorded[0]
    assert fitted.idata.posterior.attrs["map_failed_starts"] == failed
