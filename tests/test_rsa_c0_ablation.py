"""Ablating chain 0's final: the switchable rewrite is the final, and each switch does what it says."""

import dataclasses

import numpy as np
import pytest
from pyprojroot import here

from src.rsa.ablation import spec_c0 as S
from src.rsa.ablation.variants import VARIANTS as V
from src.rsa.context import group_by_shape
from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import RSAModel, check_contract, prior_draws

FINAL = here("data/rsa/live/finals/boundary_avoidable_ambiguity_listener.py")
# Each mechanism's parameter away from its neutral value: the gates at even odds, the leak open.
ACTIVE = {**{k: 1.5 for k in S.NEUTRAL}, "logit_p_base": 0.0, "logit_p_outlier": 0.0, "w_inhibit_leak": 1.0}


@pytest.fixture(scope="module")
def sample():
    """Plain displays, half of them given a valenced framing and some greyscale objects: the plain
    pool has one framing and no greyscale, so ``valence`` and ``color`` could not act on it."""
    pool = plain_pool()
    rng = np.random.default_rng(0)
    picked = [pool[i] for i in rng.choice(len(pool), 120, replace=False)]
    varied = [dataclasses.replace(c, valence=1 if i % 2 else -1,
                                  grayscale=tuple(j % 2 for j in range(len(c.objects))))
              for i, c in enumerate(picked[:60])]
    return group_by_shape(varied + picked[60:])


def variant(mechanisms, tmp_path):
    name, path = V.write_variant("c0", mechanisms, tmp_path)
    return RSAModel(path, name=name)


def predictions(model, params, sample):
    return {shape: np.asarray(model.group_probs({k: params[k] for k in model.params}, g))
            for shape, g in sample.items()}


def max_difference(a, b):
    return max(float(np.max(np.abs(a[s] - b[s]))) for s in a)


def _prior(d):
    return type(d).__name__, {k: float(getattr(d, k)) for k in d.arg_constraints}


def test_with_every_mechanism_on_a_variant_is_the_final(sample, tmp_path):
    full = variant(S.MECHANISMS, tmp_path)
    final = RSAModel(FINAL)
    assert full.name == "c0_full"
    assert list(full.params) == list(final.params)
    assert all(_prior(full.params[k]) == _prior(final.params[k]) for k in final.params)
    for draw in prior_draws(final.params, 4, seed=0):
        assert max_difference(predictions(full, draw, sample), predictions(final, draw, sample)) < 1e-6


def test_names_requirements_and_sizes():
    assert V.variant_name("c0", set(S.MECHANISMS) - {"color", "outlier"}) == "c0_minus_col_out"
    assert "attention" in V.invalid_reason("c0", ["popout"])
    assert "attention" not in V.removals("c0", S.MECHANISMS)  # its four terms need it
    assert "attention" in V.removals("c0", set(S.MECHANISMS) - {"popout", "load", "candidate", "inhibit_leak"})
    assert V.size("c0", S.MECHANISMS) == {"population": 17, "person_vectors": 0}
    assert V.size("c0", ()) == {"population": 2, "person_vectors": 0}


@pytest.mark.parametrize("mechanism", [m for m in S.MECHANISMS if S.PARAMETERS[m]])
def test_a_mechanism_at_its_neutral_value_is_the_model_without_it(mechanism, sample, tmp_path):
    needing_it = {m for m, needs in S.REQUIRES.items() if mechanism in needs}
    on = [m for m in S.MECHANISMS if m not in needing_it]
    with_it, without = variant(on, tmp_path), variant(set(on) - {mechanism}, tmp_path)
    assert set(with_it.params) - set(without.params) == set(S.PARAMETERS[mechanism])
    assert len(with_it.params) == V.size("c0", on)["population"]
    draw = {k: float(v) for k, v in prior_draws(with_it.params, 1, seed=1)[0].items()}
    draw.update({k: S.NEUTRAL[k] for k in S.PARAMETERS[mechanism]})
    assert max_difference(predictions(with_it, draw, sample), predictions(without, draw, sample)) < 1e-6


def _effect(mechanism, sample, tmp_path):
    """How far switching ``mechanism`` on (away from its neutral value) moves any prediction."""
    needing_it = {m for m, needs in S.REQUIRES.items() if mechanism in needs}
    on = [m for m in S.MECHANISMS if m not in needing_it]
    with_it, without = variant(on, tmp_path), variant(set(on) - {mechanism}, tmp_path)
    draw = {k: float(v) for k, v in prior_draws(with_it.params, 1, seed=2)[0].items()}
    draw.update({k: ACTIVE[k] for k in S.PARAMETERS[mechanism]})
    return max_difference(predictions(with_it, draw, sample), predictions(without, draw, sample))


@pytest.mark.parametrize("mechanism", [m for m in S.MECHANISMS if m not in S.INERT])
def test_each_mechanism_moves_some_prediction_when_on(mechanism, sample, tmp_path):
    """A switch that does nothing would be dropped for free: each must move some display."""
    assert _effect(mechanism, sample, tmp_path) > 1e-3


@pytest.mark.parametrize("mechanism", S.INERT)
def test_an_inert_mechanism_of_the_final_moves_nothing(mechanism, sample, tmp_path):
    """popout raises attention only when the word fits one object, where attending and the
    captured inattentive listener choose that same object: the final carries it to no effect."""
    assert _effect(mechanism, sample, tmp_path) < 1e-5


def test_the_full_and_the_core_variants_meet_the_model_contract(sample, tmp_path):
    for mechanisms in (S.MECHANISMS, ()):
        check_contract(variant(mechanisms, tmp_path), sample)


def _small_data(tmp_path):
    import pandas as pd

    from src.rsa.dataset import DEFAULT_TRIALS_CSV

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "pooled.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].assign(source="pragmods").to_csv(data, index=False)
    return data


def _answer(args, doc, worth, converged=True, seed=0):
    """Fabricated CVs for the models ``doc`` asks for: each mechanism on is worth ``worth`` nats."""
    from src.rsa.consolidate.eliminate import read_folds

    rng = np.random.default_rng(seed)
    n = len(read_folds(args.data, args.out / "folds", args.folds, args.fold_seed).fold)
    for ms, name in zip(doc["sets"], doc["names"]):
        total = -1000.0 + sum(worth[m] for m in ms)
        noise = rng.normal(0, 0.1, n)
        np.savez(args.out / "cv" / f"{name}.npz", pointwise=np.full(n, total / n) + noise - noise.mean(),
                 converged=converged)


def test_plan_replays_the_rule_round_by_round_over_saved_cvs(tmp_path):
    """No fits: interior is worth 400 nats, the rest nothing; popout must go before attention."""
    import json

    from src.rsa.ablation import eliminate as elim

    args = elim.Args(data=_small_data(tmp_path), out=tmp_path / "e", cache=tmp_path / "c", folds=2,
                     start=["interior", "attention", "popout", "outlier"])
    worth = {"interior": 400.0, "attention": 0.0, "popout": 0.0, "outlier": 0.0}
    for round_ in range(10):
        doc = elim.plan(args)
        if doc["status"] == "done":
            break
        assert doc["attempt"] == 1
        _answer(args, doc, worth, seed=round_)
    assert doc["lean"]["mechanisms"] == ["interior"]
    record = json.loads((args.out / "elimination.json").read_text())
    dropped = [r["decision"].removeprefix("drop ") for r in record["rounds"] if r["decision"].startswith("drop")]
    assert sorted(dropped) == ["attention", "outlier", "popout"]
    assert dropped.index("popout") < dropped.index("attention")
    assert record["config"]["fit"]["init"] == "map"
    assert (args.out / "lean" / "c0_minus_d2_avo_conf_sing_dist_feat_val_col_typ_att_pop_load_cand_leak_out.py").exists()


def test_an_unconverged_starting_model_stops_the_elimination(tmp_path):
    from src.rsa.ablation import eliminate as elim

    args = elim.Args(data=_small_data(tmp_path), out=tmp_path / "e", cache=tmp_path / "c", folds=2,
                     start=["interior", "outlier"])
    _answer(args, elim.plan(args), {"interior": 400.0, "outlier": 0.0}, converged=False)
    with pytest.raises(RuntimeError, match="did not converge on every fold"):
        elim.plan(args)
