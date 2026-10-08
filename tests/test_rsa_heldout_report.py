"""The held-out condition-by-condition page of a brought-back cell."""

import hashlib
import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.evaluate_heldout import Args as HeldoutArgs, main as evaluate
from src.rsa.fit import FitSettings
from src.rsa.loop.cv import cv_pointwise, make_folds
from src.rsa.heldout_report import Args, chosen_models, main
from src.rsa.split import split
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
QUICK = dict(num_warmup=200, num_samples=200, num_chains=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    """A cell as Sherlock brings it back: exported rsa_l1, literal_listener pruned."""
    tmp = tmp_path_factory.mktemp("heldout_report")
    df = pd.read_csv(DEFAULT_TRIALS_CSV).assign(source="pragmods")
    train, test, _ = split(df[df.experiment.isin(["E8_levels", "E9_twins", "E7_color"])], 0.25, seed=3)
    cell = tmp / "real_rep1"
    (cell / "models" / "pruned").mkdir(parents=True)
    train_path, test_path = tmp / "train.csv", tmp / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    train.to_csv(cell / "responses.csv", index=False)
    shutil.copyfile(SEEDS / "rsa_l1.py", cell / "models" / "rsa_l1.py")
    shutil.copyfile(SEEDS / "literal_listener.py", cell / "models" / "pruned" / "literal_listener.py")
    (cell / "models" / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "rsa_l1"}]}))
    seeds = tmp / "seeds"
    seeds.mkdir()
    shutil.copyfile(SEEDS / "rsa_l2.py", seeds / "rsa_l2.py")
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "rsa_l2"}]}))
    evaluate(HeldoutArgs(loop_dir=cell, test=test_path, out_dir=cell / "heldout", seed_models=seeds, **QUICK))
    (cell / "export.json").write_text(json.dumps({"best_model": "rsa_l1", "live": ["rsa_l1"]}))
    (cell / "data.sha256").write_text(f"{sha(train_path)}  train.csv\n{sha(test_path)}  test.csv\n")
    # The loop's grouped CV of the models the page shows, as the loop leaves it.
    settings = FitSettings(**json.loads((cell / "heldout" / "heldout.json").read_text())["settings"])
    folds = make_folds(cell / "responses.csv", cell / ".cv", 2, seed=settings.seed)
    for path in (seeds / "rsa_l2.py", cell / "models" / "rsa_l1.py", cell / "models" / "pruned" / "literal_listener.py"):
        cv_pointwise(path, path.stem, cell / "responses.csv", folds, settings, cell / ".fit_cache")
    return cell, train_path, test_path, tmp


def test_the_page_draws_the_held_out_conditions_for_seed_exported_and_best(cell, monkeypatch):
    cell, train, test, tmp = cell
    monkeypatch.setattr("src.rsa.heldout_report.SEED_DIR", tmp / "seeds")
    out = main(Args(cell=cell, train=train, test=test, work=tmp / "work", top=1))
    bundle = json.loads(out.with_suffix(".bundle.json").read_text())
    names = {m["name"] for m in bundle["models"]}
    assert {"rsa_l2 (seed)", "rsa_l1 (exported)", "literal_listener (pruned)"} == names
    assert bundle["metric"]["diff_label"] == "Δlpd"
    seed = next(m for m in bundle["models"] if m["name"] == "rsa_l2 (seed)")
    assert seed["heldout_vs_seed"] == 0.0
    # Only the held-out displays are drawn, with their observed counts.
    test_trials = load_forced_choice(test)
    assert sum(p["n"] for p in bundle["panels"]) == len(test_trials.contexts)
    assert "__BUNDLE_JSON__" not in out.read_text()
    # Per-unit lpd sums, held out and in sample, for every shown model.
    units = pd.read_csv(cell / "heldout" / "unit_lpd.csv")
    assert set(units.split) == {"test", "train"} and set(units.model) == names
    shown = units[(units.split == "test") & (units.model == "rsa_l2 (seed)")]
    assert shown.n.sum() == len(test_trials.contexts)
    assert shown.lpd.sum() == pytest.approx(next(m for m in bundle["models"] if m["name"] == "rsa_l2 (seed)")["elpd_loo"])


def test_from_the_loop_directory_the_evaluations_fits_are_reused(cell, monkeypatch):
    cell, train, test, tmp = cell
    monkeypatch.setattr("src.rsa.heldout_report.SEED_DIR", tmp / "seeds")
    before = sorted(p.name for p in (cell / ".fit_cache").glob("*.nc"))
    out = main(Args(cell=cell, loop_dir=cell, test=test, work=tmp / "work_loop", top=1))
    assert sorted(p.name for p in (cell / ".fit_cache").glob("*.nc")) == before  # nothing refitted
    heldout = json.loads((cell / "heldout" / "heldout.json").read_text())
    recorded = {r["model"]: r["lpd"] for r in heldout["table"]}
    bundle = json.loads(out.with_suffix(".bundle.json").read_text())
    exported = next(m for m in bundle["models"] if m["name"] == "rsa_l1 (exported)")
    assert exported["elpd_loo"] == pytest.approx(recorded["rsa_l1"], abs=1e-6)
    # From the loop directory, also each training unit's out-of-fold lpd.
    units = pd.read_csv(cell / "heldout" / "unit_lpd.csv")
    assert set(units.split) == {"test", "train", "cv"}
    cv = units[units.split == "cv"].groupby("model").lpd.sum()
    train = units[units.split == "train"].groupby("model").lpd.sum()
    assert (cv < train).all()  # out of fold predicts worse than in sample


def test_train_and_loop_dir_are_exclusive(cell):
    cell, train, test, tmp = cell
    with pytest.raises(ValueError, match="exactly one"):
        main(Args(cell=cell, train=train, loop_dir=cell, test=test, work=tmp / "w"))


def test_a_held_out_label_must_be_in_the_cells_table(cell):
    cell, *_ = cell
    heldout = json.loads((cell / "heldout" / "heldout.json").read_text())
    with pytest.raises(ValueError, match="not in"):
        chosen_models(cell, heldout, 0, ["pruned:no_such_model"])


def test_data_that_is_not_the_cells_refuses(cell):
    cell, train, test, tmp = cell
    other = tmp / "other_test.csv"
    pd.read_csv(test).iloc[:-1].to_csv(other, index=False)
    with pytest.raises(ValueError, match="is not data"):
        main(Args(cell=cell, train=train, test=other, work=tmp / "work"))
