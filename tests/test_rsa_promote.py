"""Promoting run 2's models to seed the live experiments (PI decision 2026-10-08)."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.promote import Args, candidates, cluster, main
from src.rsa.split import split
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def make_cell(root, cell, live, pruned):
    """A brought-back cell: its seeds, admitted models (live or pruned) and ledger."""
    cdir = root / cell
    (cdir / "models" / "pruned").mkdir(parents=True)
    history = [dict(step=0, round=-1, best_model="rsa_l1",
                    events=[dict(outcome="seeded", name="rsa_l1"), dict(outcome="seeded", name="literal_listener")])]
    (cdir / "history.json").write_text(json.dumps(history))
    ledger = []
    for name, src in live.items():
        shutil.copyfile(SEEDS / f"{src}.py", cdir / "models" / f"{name}.py")
        ledger.append(dict(name=name, outcome="admitted", detail="", hypothesis=src, context="round 1"))
    for name, src in pruned.items():
        shutil.copyfile(SEEDS / f"{src}.py", cdir / "models" / "pruned" / f"{name}.py")
        ledger += [dict(name=name, outcome="admitted", detail="", hypothesis=src, context="round 1"),
                   dict(name=name, outcome="pruned", detail="behind", hypothesis=src, context="round 1")]
    # A seed is in the set but never a candidate.
    shutil.copyfile(SEEDS / "rsa_l1.py", cdir / "models" / "rsa_l1.py")
    (cdir / "attempted_hypotheses.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))


@pytest.fixture
def sweep(tmp_path):
    root = tmp_path / "sweep"
    make_cell(root, "real_rep1", live={"deep": "rsa_l2", "salient": "rsa_l1_salience"}, pruned={"plain": "literal_listener"})
    # The same source under another name in another cell counts once.
    make_cell(root, "real_rep2", live={"deep_again": "rsa_l2"}, pruned={"shared": "rsa_l1_shared_prior"})
    return root


def test_candidates_are_every_admitted_model_once_and_never_a_seed(sweep):
    cands = candidates(sweep, ["real_rep1", "real_rep2"])
    assert [c.key for c in cands] == ["real_rep1/deep", "real_rep1/salient", "real_rep1/plain", "real_rep2/shared"]
    assert {c.key: c.status for c in cands}["real_rep1/plain"] == "pruned"


def test_cluster_cuts_into_k_groups_by_prediction_distance():
    import numpy as np

    preds = {"a": np.array([0.1, 0.9]), "a2": np.array([0.11, 0.89]), "b": np.array([0.8, 0.2]), "c": np.array([0.5, 0.5])}
    groups = cluster(preds, 3)
    assert groups["a"] == groups["a2"] and len(set(groups.values())) == 3


@pytest.mark.slow  # runs NUTS
def test_one_model_per_group_by_grouped_cv_plus_the_reference(sweep, tmp_path):
    df = pd.read_csv(DEFAULT_TRIALS_CSV).assign(source="pragmods")
    train, test, _ = split(df[df.experiment.isin(["E8_levels", "E5_baserate", "E9_twins"])], 0.25, seed=1)
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "live_seeds"
    record = main(Args(sweep=sweep, train=tmp_path / "train.csv", test=tmp_path / "test.csv", work=tmp_path / "work",
                       out=out, k=2, cells=["real_rep1", "real_rep2"], folds=2, num_warmup=200, num_samples=200,
                       num_chains=2, workers=2))
    assert len(record["candidates"]) == 4 and len(record["groups"]) == 2
    # Each group's chosen model has the best grouped CV of its members.
    cv = {c["key"]: c["elpd_cv"] for c in record["candidates"]}
    for g in record["groups"]:
        assert cv[g["chosen"]] == max(cv[m] for m in g["members"])
    manifest = yaml.safe_load((out / "models" / "models_manifest.yaml").read_text())["models"]
    names = [m["name"] for m in manifest]
    assert len(names) == 4 and names[-2:] == ["rsa_l2", "rsa_l1"] and record["promoted"] == names
    assert all((out / "models" / f"{n}.py").exists() for n in names)
    # The combined data holds every training and held-out trial.
    assert record["data_sha256"].keys() == {"train.csv", "test.csv"}
