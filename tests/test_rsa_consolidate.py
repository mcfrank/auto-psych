"""Consolidating the three finals: the superset's switches, the elimination rule, the scoring."""

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from src.rsa.consolidate import eliminate as elim
from src.rsa.consolidate.superset import MECHANISM_PARAMS, MECHANISMS, OFF, make
from src.rsa.consolidate.variants import canonical, variant_name, write_variant
from src.rsa.context import group_by_shape
from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import RSAModel, check_contract
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@pytest.fixture(scope="module")
def sample():
    pool = plain_pool()
    rng = np.random.default_rng(0)
    return group_by_shape([pool[i] for i in rng.choice(len(pool), 150, replace=False)])


def probs(fn, params, groups):
    import jax

    return {s: np.asarray(jax.vmap(lambda c: fn(params, c))(g.arrays)) for s, g in groups.items()}


def test_with_nothing_on_the_superset_is_the_rsa_l1_starting_model(sample, tmp_path):
    _, core = make(())
    seed = RSAModel(SEEDS / "rsa_l1.py")
    p = {"alpha": 2.3, "lapse": 0.07}
    mine = probs(core, p, sample)
    for shape, g in sample.items():
        theirs = np.asarray(seed.group_probs(p, g))
        assert np.allclose(mine[shape], theirs, atol=1e-5), shape


def test_every_mechanism_at_its_off_value_is_the_model_without_it(sample):
    """The full model with every parameter at its neutral value equals the depth-2
    core (attention aside: it is off by structure, being a sigmoid)."""
    on = [m for m in MECHANISMS if m != "attention"]
    params, full = make(on)
    _, core2 = make(["depth2"])
    p = {k: OFF.get(k, 0.0) for k in params}
    p.update(alpha=1.7, lapse=0.05)
    a, b = probs(full, p, sample), probs(core2, {"alpha": 1.7, "lapse": 0.05}, sample)
    for shape in sample:
        assert np.allclose(a[shape], b[shape], atol=1e-5), shape


@pytest.mark.parametrize("mechanism", [m for m in MECHANISMS if m != "depth2"])
def test_each_mechanism_changes_predictions_when_on(mechanism, sample):
    """A switch that does nothing would be dropped for free: each must move some display."""
    params, fn = make(["depth2", mechanism])
    _, base = make(["depth2"])
    p = {"alpha": 1.7, "lapse": 0.05}
    p.update({k: 0.8 for k in MECHANISM_PARAMS[mechanism]})
    if mechanism == "attention":
        p.update(logit_p_attend=0.0, w_load=0.5)
    a, b = probs(fn, p, sample), probs(base, {"alpha": 1.7, "lapse": 0.05}, sample)
    assert max(np.abs(a[s] - b[s]).max() for s in sample) > 1e-3


def test_variant_files_are_self_contained_named_by_mechanisms_and_meet_the_contract(sample, tmp_path):
    name, path = write_variant(["confusion", "dup", "depth2"], tmp_path)
    assert name == "cons_d2_dup_conf" == variant_name(["depth2", "confusion", "dup"])
    assert "def make(" in path.read_text() and "MECHANISMS_ON = ('depth2', 'dup', 'confusion')" in path.read_text()
    check_contract(RSAModel(path, name=name), sample)
    full_name, full = write_variant(MECHANISMS, tmp_path)
    check_contract(RSAModel(full, name=full_name), sample)
    assert len(RSAModel(full).params) == 16
    with pytest.raises(ValueError, match="unknown mechanisms"):
        canonical(["dup", "telepathy"])


def scored(elpd_by_set, n=40, seed=0):
    """Fake CV: each mechanism set's pointwise lpd, its total as given, small noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for ms, total in elpd_by_set.items():
        noise = rng.normal(0, 0.1, n)  # each model its own: differences have a nonzero SE
        out[canonical(ms)] = elim.Scored(np.full(n, total / n) + noise - noise.mean(), True)
    return out


def test_elimination_drops_free_mechanisms_and_stops_at_a_costly_one():
    """dup and conf cost nothing; aspect costs 10 nats per unit (far beyond 2 SE)."""
    def value(ms):
        return -1000.0 + (400.0 if "aspect" in ms else 0.0) + (0.2 if "dup" in ms else 0.0)

    calls = []

    def score(sets):
        calls.append(sets)
        return {canonical(s): v for s, v in scored({s: value(s) for s in sets}, seed=len(calls)).items()}

    units = np.repeat(np.arange(20), 2)
    rec = elim.eliminate(("dup", "aspect", "confusion"), score, units)
    assert [r["decision"].split(":")[0] for r in rec["rounds"]] == ["drop confusion", "drop dup", "stop"]
    assert rec["lean"]["mechanisms"] == ["aspect"]
    assert "aspect" in rec["rounds"][-1]["decision"]


def test_elimination_never_drops_into_an_unconverged_model():
    def score(sets):
        out = scored({s: -100.0 for s in sets})
        return {k: elim.Scored(v.pointwise, converged=k == canonical(("dup", "aspect"))) for k, v in out.items()}

    rec = elim.eliminate(("dup", "aspect"), score, np.arange(40))
    assert rec["rounds"][0]["decision"] == "stop: no one-off model converged"
    assert rec["lean"]["mechanisms"] == ["dup", "aspect"]


@pytest.mark.slow  # runs NUTS
def test_elimination_and_scoring_run_end_to_end_on_a_small_dataset(tmp_path):
    from src.rsa.consolidate import score
    from src.rsa.dataset import DEFAULT_TRIALS_CSV

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "pooled.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].assign(source="pragmods").to_csv(data, index=False)
    fit = dict(num_warmup=100, num_samples=100, num_chains=2)
    rec = elim.main(elim.Args(data=data, out=tmp_path / "elim", cache=tmp_path / "cache", workers=4, folds=2,
                              start=["dup", "depth2"], **fit))
    assert rec["rounds"] and rec["lean"]["model"].startswith("cons_")
    assert (tmp_path / "elim" / "lean" / f"{rec['lean']['model']}.py").exists()
    saved = json.loads((tmp_path / "elim" / "elimination.json").read_text())
    assert saved["data_sha256"] and saved["rule"]["threshold_dse"] == 2.0
    # Resuming reuses every saved CV (no refit): same record.
    again = elim.main(elim.Args(data=data, out=tmp_path / "elim", cache=tmp_path / "cache", workers=4, folds=2,
                                start=["dup", "depth2"], **fit))
    assert again["lean"] == rec["lean"]

    test = tmp_path / "test.csv"
    t = df[df.experiment.isin(["E9_twins"])].assign(source="pragmods")
    t.to_csv(test, index=False)
    lean = tmp_path / "elim" / "lean" / f"{rec['lean']['model']}.py"
    out = score.main(score.Args(train=data, test=test, cache=tmp_path / "cache",
                                models=[f"lean={lean}", f"l1={SEEDS / 'rsa_l1.py'}"], reference="l1",
                                out=tmp_path / "score.json", workers=2, **fit))
    assert set(out["models"]) == {"lean", "l1"} and "vs_reference" in out["models"]["lean"]
