"""Held-out evaluation of loop models."""

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.evaluate_heldout import Args, heldout_lpd, main
from src.rsa.fit import FitSettings, fit
from src.rsa.model_file import RSAModel
from src.rsa.split import split
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
QUICK = dict(num_warmup=300, num_samples=300, num_chains=2)


def manifest(d, names):
    (d / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": n, "rationale": n} for n in names]}))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("heldout")
    df = pd.read_csv(DEFAULT_TRIALS_CSV).assign(source="pragmods")
    df = df[df.experiment.isin(["E8_levels", "E9_twins", "E5_baserate", "E7_color"])]
    train, test, _ = split(df, 0.25, seed=2)
    loop = tmp / "loop"
    (loop / "models").mkdir(parents=True)
    train.to_csv(loop / "responses.csv", index=False)
    shutil.copyfile(SEEDS / "rsa_l2.py", loop / "models" / "rsa_l2.py")
    manifest(loop / "models", ["rsa_l2"])
    seeds = tmp / "seeds"
    seeds.mkdir()
    for n in ("literal_listener", "rsa_l1"):
        shutil.copyfile(SEEDS / f"{n}.py", seeds / f"{n}.py")
    manifest(seeds, ["literal_listener", "rsa_l1"])
    test_path = tmp / "test.csv"
    test.to_csv(test_path, index=False)
    return loop, seeds, test_path, tmp


def test_pragmatic_models_generalise_better_than_the_literal_listener(run):
    loop, seeds, test_path, tmp = run
    table = main(Args(loop_dir=loop, test=test_path, out_dir=tmp / "out", seed_models=seeds, **QUICK))
    lpd = dict(zip(table.model, table.lpd))
    assert lpd["seed:rsa_l1"] > lpd["seed:literal_listener"]
    assert lpd["rsa_l2"] > lpd["seed:literal_listener"]
    assert (tmp / "out" / "heldout.csv").exists() and "lpd[pragmods]" in table.columns
    # The best seed is the reference: its difference and SE are zero.
    best = table[table.model == "seed:rsa_l1"].iloc[0]
    assert best.diff_vs_best_seed == 0.0 and best.se_diff_clustered == 0.0


def test_lpd_is_the_log_of_the_posterior_mean_probability(run):
    loop, _, test_path, _ = run
    trials = load_forced_choice(loop / "responses.csv")
    model = RSAModel(SEEDS / "literal_listener.py")
    fitted = fit(model, trials.contexts, trials.choices, FitSettings(**QUICK))
    test = load_forced_choice(test_path)
    lpd = heldout_lpd(model, fitted, test.contexts[:5], test.choices[:5], max_draws=50)
    assert lpd.shape == (5,) and np.all(lpd <= 0) and np.all(np.isfinite(lpd))
