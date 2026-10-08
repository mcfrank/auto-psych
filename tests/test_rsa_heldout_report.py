"""The held-out condition-by-condition page of a brought-back cell."""

import hashlib
import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.evaluate_heldout import Args as HeldoutArgs, main as evaluate
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


def test_a_held_out_label_must_be_in_the_cells_table(cell):
    cell, *_ = cell
    heldout = json.loads((cell / "heldout" / "heldout.json").read_text())
    with pytest.raises(ValueError, match="not in"):
        chosen_models(cell, heldout, 0, ["pruned:no_such_model"])


def test_data_that_is_not_the_cells_refuses(cell):
    cell, train, test, tmp = cell
    with pytest.raises(ValueError, match="is not data"):
        main(Args(cell=cell, train=test, test=test, work=tmp / "work"))
