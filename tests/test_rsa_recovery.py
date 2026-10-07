"""Simulated participants, the recovery run's seed pool, and the recovery verdict."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.loop.run import seed_pool
from src.rsa.split import split
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
QUICK = dict(num_warmup=300, num_samples=300, num_chains=2)


def test_the_seed_pool_leaves_out_the_ground_truth(tmp_path):
    pool = seed_pool(SEEDS, ["literal_listener"], tmp_path)
    names = [e["name"] for e in yaml.safe_load((pool / "models_manifest.yaml").read_text())["models"]]
    assert "literal_listener" not in names and "rsa_l1" in names
    assert not (pool / "literal_listener.py").exists()
    assert seed_pool(SEEDS, [], tmp_path) == SEEDS
    with pytest.raises(ValueError, match="names no seed"):
        seed_pool(SEEDS, ["no_such_model"], tmp_path)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    from src.rsa.simulate import Args, main

    tmp = tmp_path_factory.mktemp("sim")
    df = pd.read_csv(DEFAULT_TRIALS_CSV).assign(source="pragmods")
    df = df[df.experiment.isin(["E8_levels", "E9_twins", "E5_baserate"])]
    real = tmp / "real.csv"
    df.to_csv(real, index=False)
    out, prov = tmp / "sim.csv", tmp / "hidden" / "provenance.json"
    main(Args(gt=SEEDS / "rsa_l2.py", fit_on=real, displays=real, out=out, provenance=prov, seed=5, **QUICK))
    return tmp, real, out, prov


@pytest.mark.slow
def test_simulation_replaces_only_the_choices_and_hides_the_ground_truth(simulated):
    tmp, real, out, prov = simulated
    a, b = pd.read_csv(real), pd.read_csv(out)
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    assert (a.drop(columns=["choice"]).fillna("") == b.drop(columns=["choice"]).fillna("")).all().all()
    assert (a.choice != b.choice).any()
    assert "rsa_l2" not in out.read_text()
    assert json.loads(prov.read_text())["ground_truth"].endswith("rsa_l2.py")


@pytest.mark.slow
def test_a_run_whose_best_model_is_the_ground_truth_recovers_it(simulated):
    from src.rsa.recovery import Args, main

    tmp, _, sim, _ = simulated
    train, test, _ = split(pd.read_csv(sim), 0.25, seed=1)
    loop = tmp / "loop"
    (loop / "models").mkdir(parents=True)
    train.to_csv(loop / "responses.csv", index=False)
    shutil.copyfile(SEEDS / "rsa_l2.py", loop / "models" / "found_depth_two.py")
    (loop / "models" / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "found_depth_two", "rationale": "x"}]}))
    (loop / "export.json").write_text(json.dumps({"best_model": "found_depth_two", "live": ["found_depth_two"]}))
    seeds = tmp / "seeds"
    seeds.mkdir()
    shutil.copyfile(SEEDS / "literal_listener.py", seeds / "literal_listener.py")
    (seeds / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "literal_listener", "rationale": "x"}]}))
    test_path = tmp / "test.csv"
    test.to_csv(test_path, index=False)
    verdict = main(Args(loop_dir=loop, gt=SEEDS / "rsa_l2.py", test=test_path, out_dir=tmp / "rec",
                        seed_models=seeds, **QUICK))
    assert verdict["recovered"]
    assert verdict["best_pool_rmse_to_gt"] < 0.01
