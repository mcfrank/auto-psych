"""Designing a live experiment from fitted memo models, end to end."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.design.run import Args, class_layout, design_pool, kinds, main, parse_quotas
from src.rsa.experiment.design import Design, trial_list
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def test_the_pool_is_what_the_page_shows_with_one_slot_per_choice_class():
    pool = design_pool()
    valid = class_layout(pool)
    assert len(pool) == 1597 and valid.shape == (1597, 4)  # 794 before redundant features and 2 x 3/4 (2026-10-10)
    twins = next(i for i, c in enumerate(pool) if len(set(c.objects)) < len(c.objects))
    assert valid[twins].sum() == len(set(pool[twins].objects))
    assert all(c.familiarization is None and c.grayscale is None for c in pool)


def test_quotas_are_shares_of_the_design_rounded_half_up():
    pool = design_pool()
    q2, q3, prior = parse_quotas(["objects=2:0.15", "objects=3:0.25", "query=prior:0.2"], pool, 40)
    assert (q2.minimum, q3.minimum, prior.minimum) == (6, 10, 8)
    assert q2.dimension == q3.dimension == "objects" and prior.dimension == "query"
    assert q2.members.sum() == sum(len(c.objects) == 2 for c in pool) == 54
    assert parse_quotas(["objects=2:0.15"], pool, 10)[0].minimum == 2  # 1.5 -> 2
    for bad in ["objects=5:0.1", "colour=red:0.1", "objects=2", "objects=2:1.5"]:
        with pytest.raises(ValueError):
            parse_quotas([bad], pool, 40)
    assert kinds(pool, [0, 1]) == {"2x2 prior": 1, "2x2 word": 1}  # ((0, 0), (1, 1)): one word (its two features are synonyms), then mumble


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


@pytest.mark.slow  # runs NUTS
def test_the_bar_models_join_the_design_merged_where_they_duplicate_and_with_their_share_of_the_prior(tmp_path):
    def folder(name, models):
        d = tmp_path / name
        d.mkdir()
        for n in models:
            shutil.copyfile(SEEDS / f"{n}.py", d / f"{n}.py")
        (d / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": n} for n in models]}))
        return d

    carried = folder("carried", ["rsa_l1", "rsa_l2"])
    bar = folder("bar", ["literal_listener", "rsa_l1", "rsa_l1_salience"])
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "trials.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].to_csv(data, index=False)
    record = main(Args(models_dir=carried, bar_models_dirs=[bar], data=data, cache=tmp_path / "cache",
                       out=tmp_path / "out", trials_per_participant=4, participants=40, displays=[8],
                       power_participants=[40], n_draws=20, n_scenarios=300, n_power_scenarios=300,
                       num_warmup=200, num_samples=200, num_chains=2,
                       quotas=["objects=2:0.25", "query=prior:0.25"]))
    # The quotas are met, and the free design is beside it for the comparison.
    (design,) = record["designs"]
    assert [(q["kind"], q["minimum"]) for q in design["quotas"]] == [("objects=2", 2), ("query=prior", 2)]
    assert all(q["count"] >= q["minimum"] for q in design["quotas"])
    assert sum(design["kinds"].values()) == 8 == len(design["free"]["picks"])
    assert design["free"]["power"][0]["participants"] == 40 and set(design["free"]["quota_counts"]) == {
        "objects=2", "query=prior"}
    # The bar's rsa_l1 is the carried file: merged, not designed for twice.
    assert {"model": "bar:rsa_l1", "same_as": "rsa_l1", "reason": "the same model file"} in record["merged"]
    assert record["models"][:2] == ["rsa_l1", "rsa_l2"] and "bar:literal_listener" in record["models"]
    groups = record["model_groups"]
    prior = dict(zip(record["models"], record["prior"]))
    assert prior["rsa_l1"] == pytest.approx(0.25)  # half the mass over the two carried models
    assert sum(p for p, g in zip(record["prior"], groups) if g == "bar") == pytest.approx(0.5)
    assert set(record["designs"][0]["power"][0]["by_model"]) == set(record["models"])
