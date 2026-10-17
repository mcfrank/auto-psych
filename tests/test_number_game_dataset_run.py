"""A number-game dataset run: the live pipeline's stages, collected from the dataset.

``data_source: dataset`` replaces the ground truth: every experiment is
designed as in a live run (on the cells with ``n_participants`` responses
left), collected from the Bigelow & Piantadosi dataset without reusing a
response, and modelled by the agent loop, which starts from every seed. There
is no ground truth, so nothing is scored on data the loop never collected: the
result records what the loop concluded (per step, its best model).
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from src.number_game.dataset import DATA_CSV, DatasetCollection
from src.runtime import coding_agent
from src.runtime.config import REPO_ROOT
from src.subjective_randomness.holdout_recovery import run_holdout_recovery_from_config
from src.tasks import number_game

SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
SEEDS = ["tenenbaum_rules_similarity", "bigelow_piantadosi_independent", "bigelow_piantadosi_interval"]


def _scripted_agent(run, *, allowed_dirs, usage_label, **kwargs):
    if usage_label == "inner:candidate":
        out = Path(allowed_dirs[0])
        shutil.copyfile(SEEDS_DIR / "bigelow_piantadosi_independent.py", out / "candidate.py")
        (out / "hypothesis.md").write_text("Rules weighted one by one, and intervals.\n")
        (out / "model_name.txt").write_text("weighted_rules\n")
    return True, None


@pytest.mark.slow
@pytest.mark.skipif(not DATA_CSV.exists(), reason="number-game dataset not present")
def test_a_dataset_run_collects_real_responses_once_and_records_the_loops_conclusions(tmp_path, monkeypatch):
    monkeypatch.setattr(coding_agent, "run_coding_agent", _scripted_agent)
    sets = number_game.number_sets()[:20]
    monkeypatch.setattr(number_game, "number_sets", lambda: sets)
    config = {
        "project_id": "number_game",
        "seed_models_dir": str(SEEDS_DIR),
        "data_source": "dataset",
        "n_experiments": 2,
        "n_participants": 9,  # responses per designed cell
        "seed": 3,
        "hide_participant_id": True,
        "design": {"n_eig": 6},
        "inner_loop": {
            "max_iterations": 1, "candidate_count": 1, "n_critique_proposals": 1,
            "selection": {"criterion": "grouped_cv", "cv_group": ["number_set"], "cv_folds": 2},
        },
        "agent": {"timeout_sec": 60, "backend": None, "model": "scripted"},
        "fit": {"draws": 200, "tune": 200, "chains": 2, "method": "warm_start_nuts"},
    }
    result = run_holdout_recovery_from_config(
        config, tmp_path / "config.yaml", tmp_path / "runs",
        cache_dir=tmp_path / "cache", summary_root=tmp_path / "summaries",
    )

    [run] = result["gt_runs"]
    assert result["data_source"] == "dataset" and run["gt_model"] == "dataset"
    assert "baseline" not in run and "fitted_baseline" not in run  # nothing scored
    run_root = tmp_path / "runs" / "cell_1"
    starting = json.loads((run_root / "starting_models.json").read_text())["starting_models"]
    assert sorted(starting) == sorted(SEEDS)  # no seed is held out

    # Every response is a real one, and none is drawn twice.
    drawn = Counter()
    for exp in (1, 2):
        with (run_root / f"experiment{exp}" / "data" / "responses.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == ["number_set", "target", "trial_index", "rating"]
        cells = Counter((r["number_set"], r["target"]) for r in rows)
        assert set(cells.values()) == {9} and len(cells) == 6
        drawn += cells
    source = DatasetCollection(responses_per_cell=9, seed=0)
    for (number_set, target), n in drawn.items():
        assert n <= source.cell_count({"number_set": number_set, "target": target})

    # The loop's conclusions, step by step.
    assert [row["experiment"] for row in run["trajectory"]][:1] == [1]
    assert {row["best_model"] for row in run["trajectory"]} <= set(SEEDS) | {"weighted_rules"}
    assert "incumbent" in run


def test_the_cli_summarises_a_dataset_run_by_its_best_models(tmp_path, monkeypatch, capsys):
    # The CLI's summary used to read a ground truth's recovery metrics
    # (leakage, pearson_r), which a dataset run has none of.
    import importlib.util
    import sys

    path = REPO_ROOT / "scripts" / "subjective_randomness" / "holdout_recovery.py"
    spec = importlib.util.spec_from_file_location("_holdout_recovery_cli", path)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    trajectory = [{"experiment": 1, "step": 0, "best_model": "a"}, {"experiment": 1, "step": 1, "best_model": "b"}]
    result = {"data_source": "dataset", "gt_runs": [{"gt_model": "dataset", "trajectory": trajectory}]}
    monkeypatch.setattr(cli, "load_config", lambda path: {})
    monkeypatch.setattr(cli, "run_holdout_recovery_from_config", lambda *a, **k: result)

    out = tmp_path / "holdout.json"
    cli.main(cli.Args(config=tmp_path / "c.yaml", out=out, tidy_csv=tmp_path / "h.csv", figure=tmp_path / "h.png"))

    assert json.loads(out.read_text()) == result
    assert "dataset: 2 steps, final best model b" in capsys.readouterr().out
    assert not (tmp_path / "h.csv").exists() and not (tmp_path / "h.png").exists()
