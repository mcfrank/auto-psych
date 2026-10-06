"""Admission of memo candidates to the RSA loop."""

import numpy as np
import pandas as pd
import pytest

from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.fit import FitSettings
from src.rsa.loop.gates import GateConfig, admit
from src.rsa.loop.novelty import novelty_pool
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
QUICK = FitSettings(num_warmup=300, num_samples=300, num_chains=2)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gates")
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    responses = tmp / "responses.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].to_csv(responses, index=False)
    cfg = GateConfig(responses_path=responses, cache_dir=tmp / "cache", settings=QUICK,
                     novelty_threshold=0.002, time_limit_sec=None)
    training = load_forced_choice(responses).contexts
    pool = novelty_pool()
    return tmp, cfg, training, pool


def candidate(tmp, name, source, hypothesis="A hypothesis."):
    d = tmp / name
    d.mkdir()
    (d / "candidate.py").write_text(source)
    (d / "hypothesis.md").write_text(hypothesis)
    return d


def test_a_seed_is_admitted_then_its_copy_is_not_novel(setup):
    tmp, cfg, training, pool = setup
    src = (SEEDS / "rsa_l1.py").read_text()
    first = admit(candidate(tmp, "a", src), "rsa_l1", cfg=cfg, training=training, pool=pool, admitted_preds={})
    assert first.admitted, first.reason
    assert first.pool_preds.shape[0] > len(pool)
    again = admit(candidate(tmp, "b", src.replace("Vanilla", "Plain")), "rsa_l1_copy", cfg=cfg,
                  training=training, pool=pool, admitted_preds={"rsa_l1": first.pool_preds})
    assert not again.admitted
    assert "not novel" in again.reason and "rsa_l1" in again.reason


def test_a_different_mechanism_is_novel(setup):
    tmp, cfg, training, pool = setup
    l1 = admit(candidate(tmp, "c", (SEEDS / "rsa_l1.py").read_text()), "rsa_l1", cfg=cfg,
               training=training, pool=pool, admitted_preds={})
    lit = admit(candidate(tmp, "d", (SEEDS / "literal_listener.py").read_text()), "literal", cfg=cfg,
                training=training, pool=pool, admitted_preds={"rsa_l1": l1.pool_preds})
    assert lit.admitted, lit.reason


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda s: "import os\n" + s, "code gate"),
        (lambda s: s.replace("{EPS}", "EPS"), "failed to load or run"),
        (lambda s: s.replace('return with_lapse(', 'return 2.0 * with_lapse('), "sum to 1"),
        (lambda s: s.replace('with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])',
                             'jnp.where(ctx.is_prior > 0, uniform, heard) + 0.0 * params["lapse"]'),
         "probability 0"),
    ],
)
def test_broken_candidates_are_refused_with_a_reason(setup, mutate, reason):
    tmp, cfg, training, pool = setup
    src = mutate((SEEDS / "rsa_l1.py").read_text())
    out = admit(candidate(tmp, f"bad_{abs(hash(reason))}", src), "bad", cfg=cfg,
                training=training, pool=pool, admitted_preds={})
    assert not out.admitted
    assert reason in out.reason


def test_a_candidate_without_a_hypothesis_is_refused(setup):
    tmp, cfg, training, pool = setup
    d = candidate(tmp, "nohyp", (SEEDS / "rsa_l1.py").read_text(), hypothesis="  ")
    out = admit(d, "x", cfg=cfg, training=training, pool=pool, admitted_preds={})
    assert not out.admitted and "hypothesis.md" in out.reason
