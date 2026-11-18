"""Ablating run 3's number-game winner, exception_weighted_rule_rejection: its switchable rewrite and variants.

The winner on the items no run drew (Figure 2). The elimination's rule, data and CV units are the
other number-game final's (tests/test_ng_ablation.py); only the final differs.
"""

import numpy as np
import pytest
from pyprojroot import here

from src.number_game.ablation import data as ng_data
from src.number_game.ablation import variants as V

RUN1 = ng_data.run_responses("run1")

FINAL_DIR = here() / "data/results/number_game_dataset-nov26/collected/run3/number_game/experiment3/model_loop/models"
MECHANISMS = V.spec("ew").MECHANISMS


def bound(model):
    import pymc as pm

    from src.models.data_binding import extract_observed

    with model:
        pm.set_data(extract_observed(RUN1, model))
    return model


def variant_model(on, out_dir):
    from src.models.model_loading import load_pymc_model

    name, _ = V.write_variant("ew", on, out_dir)
    return bound(load_pymc_model(name, out_dir))


def jittered(model, seed):
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, 0.5, size=np.shape(v)) for k, v in model.initial_point().items()}


def p_yes(model, point):
    out = model.replace_rvs_by_values([model["p_yes"]])[0]
    fn = model.compile_fn(out, inputs=model.value_vars, on_unused_input="ignore")
    return np.asarray(fn({k: point[k] for k in (v.name for v in model.value_vars)}))


def constrained(model, point):
    """The free variables' values (constrained, by variable name) at an unconstrained point."""
    outs = model.replace_rvs_by_values(model.free_RVs)
    fn = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore")
    values = fn({v.name: point[v.name] for v in model.value_vars})
    return {rv.name: np.asarray(v, dtype="float64") for rv, v in zip(model.free_RVs, values)}


def p_yes_at(model, values):
    """p_yes with the free variables at these constrained values (boundary values allowed)."""
    import pytensor

    rvs = model.free_RVs
    stand_ins = [rv.type() for rv in rvs]
    out = pytensor.clone_replace(model["p_yes"], dict(zip(rvs, stand_ins)))
    fn = pytensor.function(stand_ins, out, on_unused_input="ignore")
    return np.asarray(fn(*[np.asarray(values[rv.name], dtype=rv.dtype) for rv in rvs]))


def test_with_every_mechanism_on_the_variant_is_the_final(tmp_path):
    from src.models.model_loading import load_pymc_model

    original = bound(load_pymc_model(V.spec("ew").FINAL, FINAL_DIR))
    mine = variant_model(MECHANISMS, tmp_path)
    assert V.variant_name("ew", MECHANISMS) == "ew_full"
    assert {(v.name, tuple(original.initial_point()[v.name].shape)) for v in original.value_vars} == \
        {(v.name, tuple(mine.initial_point()[v.name].shape)) for v in mine.value_vars}
    logp_o, logp_m = original.compile_logp(), mine.compile_logp()
    for seed in range(4):
        point = jittered(original, seed)
        assert logp_m(point) == pytest.approx(logp_o(point), rel=1e-10, abs=1e-8)
        assert np.allclose(p_yes(mine, point), p_yes(original, point), rtol=0, atol=1e-12)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_each_mechanism_at_its_neutral_value_is_the_model_without_it(mechanism, tmp_path):
    on = set(MECHANISMS)
    off = on - {mechanism}
    assert V.invalid_reason("ew", off) is None
    with_m = variant_model(on, tmp_path)
    without_m = variant_model(off, tmp_path)
    # removing a mechanism removes its parameters only
    assert {rv.name for rv in without_m.free_RVs} < {rv.name for rv in with_m.free_RVs}
    values = constrained(with_m, jittered(with_m, 0))
    neutral = {**values, **{k: np.asarray(v, dtype="float64") for k, v in V.spec("ew").NEUTRAL[mechanism].items()}}
    kept = {rv.name for rv in without_m.free_RVs}
    assert np.allclose(p_yes_at(with_m, neutral), p_yes_at(without_m, {k: v for k, v in neutral.items() if k in kept}),
                       rtol=0, atol=1e-10)
    # ... away from it, the mechanism moves some prediction; but for zero_split, which divides one
    # weight between "ends in 0" and "contains digit 0", the same numbers (10, 20, ..., 100):
    # the final's split changes no prediction, and the elimination should drop it at no cost
    moved = p_yes_at(with_m, values) - p_yes_at(without_m, {k: v for k, v in values.items() if k in kept})
    if mechanism == "zero_split":
        assert np.abs(moved).max() < 1e-12
    else:
        assert np.abs(moved).max() > 1e-4
    # ... and the model without it has a finite log density and gradient to sample
    point = without_m.initial_point()
    assert np.isfinite(without_m.compile_logp()(point))
    assert np.all(np.isfinite(without_m.compile_dlogp()(point)))


def test_every_parameter_is_one_removable_mechanism_and_one_stays():
    from src.models.model_loading import load_pymc_model

    original = load_pymc_model(V.spec("ew").FINAL, FINAL_DIR)
    assert {rv.name for rv in original.free_RVs} == {p for m in MECHANISMS for p in V.spec("ew").NEUTRAL[m]}
    assert V.removals("ew", MECHANISMS) == list(MECHANISMS)
    # a model with no parameters cannot be sampled: the last mechanism stays
    assert V.removals("ew", ["exception"]) == []
    assert V.invalid_reason("ew", []) == f"one of {list(MECHANISMS)} must stay"
    with pytest.raises(ValueError, match="unknown mechanisms"):
        V.canonical("ew", ["exception", "telepathy"])


def test_the_full_variant_has_the_finals_parameter_count():
    from src.models.model_loading import load_pymc_model

    original = load_pymc_model(V.spec("ew").FINAL, FINAL_DIR)
    point = original.initial_point()
    free = sum(int(np.prod(point[v.name].shape)) for v in original.value_vars)
    # 13 scalars and the 44 rule-set weights (43 free on the simplex)
    assert free == 56
    assert V.size("ew", MECHANISMS) == {"population": 56, "person_vectors": 0}


def test_variant_files_are_self_contained_and_named_by_what_was_removed(tmp_path):
    name, path = V.write_variant("ew", set(MECHANISMS) - {"explaining_away", "exception"}, tmp_path)
    assert name == "ew_minus_exc_pre"
    assert "def build(on):" in path.read_text() and path.read_text().endswith("model = build(MECHANISMS_ON)\n")


def test_the_full_and_a_one_mechanism_variant_meet_the_data_contract(tmp_path):
    from src.models.model_contract import model_contract_violation

    pooled = tmp_path / "responses.csv"
    ng_data.main(ng_data.Args(out=pooled))
    for on in (MECHANISMS, ["exception"]):
        name, _ = V.write_variant("ew", on, tmp_path)
        assert model_contract_violation(name, tmp_path, pooled) is None, name


def test_the_elimination_takes_this_final():
    from src.number_game.ablation.eliminate import Args

    args = Args(final="ew", data=RUN1, out=here() / "unused")
    assert args.final == "ew"
