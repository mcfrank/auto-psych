"""The within-paper held-out split."""

import pandas as pd
import pytest

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.split import _counted, multi_trial_experiments, split, unit_keys
from src.runtime.config import PROJECT_ASSETS_DIR

DATA = PROJECT_ASSETS_DIR / "rsa_reference" / "data"


@pytest.fixture(scope="module")
def combined():
    frames = [pd.read_csv(DEFAULT_TRIALS_CSV).assign(source="pragmods")]
    for name in ("mayn_demberg_2026", "sikos_2021"):
        frames.append(pd.read_csv(DATA / f"{name}_trials.csv"))
    return pd.concat(frames, ignore_index=True)


def test_multi_trial_experiments_are_found_from_the_data(combined):
    multi = multi_trial_experiments(combined)
    assert "md2026_shapes" in multi and "sequences" in multi
    assert "E8_levels" not in multi and "sikos2021_e1" not in multi


def test_each_source_holds_out_about_the_fraction_and_no_unit_straddles(combined):
    train, test, info = split(combined, 0.2, seed=1)
    assert len(train) + len(test) == len(combined)
    for src, r in info["sources"].items():
        assert 0.2 <= r["held_out_fraction"] < 0.6, (src, r)
        assert 0 < r["held_out_units"] < r["units"]
    assert not set(unit_keys(train)) & set(unit_keys(test))


def test_one_shot_sources_hold_out_whole_conditions_and_multi_trial_ones_items(combined):
    _, test, _ = split(combined, 0.2, seed=1)
    prag = test[test.source == "pragmods"]
    one_shot = prag[~prag.experiment.isin(["sequences", "speakers", "size"])]
    full = combined[combined.source == "pragmods"]
    for (exp, cond), g in one_shot.groupby(["experiment", "condition"]):
        whole = full[(full.experiment == exp) & (full.condition == cond)]
        assert len(g) == len(whole)  # the whole condition is held out
    md = test[test.source == "mayn_demberg_2026"]
    # Items are held out, not people: held-out participants also have train trials.
    assert md.participant_id.nunique() > 200


def test_the_split_is_reproducible_and_seed_dependent(combined):
    a = split(combined, 0.2, seed=3)[2]["held_out_units"]
    assert a == split(combined, 0.2, seed=3)[2]["held_out_units"]
    assert a != split(combined, 0.2, seed=4)[2]["held_out_units"]


def test_test_rows_load_as_trials(combined, tmp_path):
    from src.rsa.dataset import load_forced_choice

    _, test, _ = split(combined, 0.2, seed=1)
    path = tmp_path / "test.csv"
    test.to_csv(path, index=False)
    trials = load_forced_choice(path)
    assert len(trials.contexts) == int(_counted(test).sum())


def test_heldout_baseline_seeds_are_the_runs_own_seed_pool(tmp_path):
    """A recovery run starts without its ground truth: the held-out baseline
    must be the seeds it had, not the project's (which include the answer)."""
    from src.rsa.evaluate_heldout import SEED_DIR, Args, seed_models_dir

    loop = tmp_path / "loop"
    loop.mkdir()
    assert seed_models_dir(Args(loop_dir=loop, test=tmp_path / "t.csv", out_dir=tmp_path)) == SEED_DIR
    (loop / "seed_pool").mkdir()
    (loop / "seed_pool" / "models_manifest.yaml").write_text("models: []\n")
    assert seed_models_dir(Args(loop_dir=loop, test=tmp_path / "t.csv", out_dir=tmp_path)) == loop / "seed_pool"
    assert seed_models_dir(Args(loop_dir=loop, test=tmp_path / "t.csv", out_dir=tmp_path, seed_models=SEED_DIR)) == SEED_DIR
