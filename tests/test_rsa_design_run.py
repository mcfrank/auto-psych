"""Designing a live experiment from fitted memo models, end to end."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.design.run import Args, class_layout, design_pool, main
from src.rsa.experiment.design import Design, trial_list
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def test_the_pool_is_what_the_page_shows_with_one_slot_per_choice_class():
    pool = design_pool()
    valid = class_layout(pool)
    assert len(pool) == 794 and valid.shape == (794, 4)
    twins = next(i for i, c in enumerate(pool) if len(set(c.objects)) < len(c.objects))
    assert valid[twins].sum() == len(set(pool[twins].objects))
    assert all(c.familiarization is None and c.grayscale is None for c in pool)


@pytest.mark.slow  # runs NUTS
def test_a_design_separates_literal_from_pragmatic_listeners_and_builds(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    names = ["literal_listener", "rsa_l1", "rsa_l2"]
    for n in names:
        shutil.copyfile(SEEDS / f"{n}.py", models / f"{n}.py")
    (models / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": n} for n in names]}))
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "trials.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].to_csv(data, index=False)
    record = main(Args(models_dir=models, data=data, cache=tmp_path / "cache", out=tmp_path / "out",
                       trials_per_participant=4, participants=40, displays=[4, 8], power_participants=[2, 60],
                       n_draws=20, n_scenarios=300, n_power_scenarios=600, num_warmup=200, num_samples=200,
                       num_chains=2))
    assert record["models"] == names and record["screened_out"] == []
    small, large = record["designs"]
    assert (small["displays"], small["responses_per_display"]) == (4, 40)  # 40 people x 4 trials / 4 displays
    assert (large["displays"], large["responses_per_display"]) == (8, 20)
    assert len(large["picks"]) == 8 and large["n_eig_picks"] >= 1
    # The first pick tells a literal listener from a pragmatic one: a word, not a prior query.
    assert small["picks"][0]["display"]["utterance"] is not None
    few, many = small["power"]
    assert (few["participants"], many["participants"]) == (2, 60)
    assert many["p_correct"] > few["p_correct"] and many["joint_eig_bits"] > few["joint_eig_bits"]
    # A design builds balanced 4-trial lists.
    design = Design.load(tmp_path / "out" / "design_d8.json")
    assert design.sha256 == large["design_sha256"]
    lst = trial_list(design, seed=1, list_index=0, n_catch=2, n_trials=4)
    assert sum(not t["is_catch"] for t in lst["trials"] if t["phase"] == "test") == 4
    assert json.loads((tmp_path / "out" / "eig.json").read_text())["trials_per_participant"] == 4
