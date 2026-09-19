"""Coverage for the recovery-ceiling analysis.

The pure parts (cell discovery, training-CSV resolution, the final-step reader,
the summary) are unit-tested on fixtures. The end-to-end path is exercised
against a **real archived cell** and marked ``slow`` — every analysis tool in
this campaign that was only ever run inside a Slurm job failed inside a Slurm
job, four times over, so this one is run against real data before it ships.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.subjective_randomness.cell_archive import CellArchiveManager
from src.subjective_randomness.recovery_ceiling import (
    CellCeiling,
    CeilingRun,
    ceiling_for_cell,
    discover_cells,
    final_step_from_csv,
    summarize_by_ground_truth,
    training_responses,
)

SWEEP_ROOT = Path(
    os.environ.get(
        "CEILING_TEST_SWEEP",
        f"{os.environ.get('SCRATCH', '/scratch')}/auto-psych/consolidation_2026_09/sweep_rerun",
    )
)


# --- pure parts ---------------------------------------------------------------


def _make_cell(root: Path, run: str, gt: str) -> Path:
    cell = root / run / gt
    (cell).mkdir(parents=True)
    (cell / "holdout.json").write_text("{}", encoding="utf-8")
    return cell


def test_discover_cells_finds_every_cell_and_raises_on_an_empty_root(tmp_path):
    _make_cell(tmp_path, "run1", "motif_stack")
    _make_cell(tmp_path, "run2", "falk_konold_dp")
    found = discover_cells(tmp_path)
    assert [f"{p.parent.parent.name}/{p.parent.name}" for p in found] == [
        "run1/motif_stack",
        "run2/falk_konold_dp",
    ]
    with pytest.raises(FileNotFoundError, match="No cells matching"):
        discover_cells(tmp_path / "run1")
    with pytest.raises(FileNotFoundError, match="No sweep root"):
        discover_cells(tmp_path / "absent")


def test_training_responses_prefers_the_last_experiment_then_pooled(tmp_path):
    run_root = tmp_path / "_runs" / "gt"
    for exp in (1, 2):
        (run_root / f"experiment{exp}" / "model_loop").mkdir(parents=True)
        (run_root / f"experiment{exp}" / "model_loop" / "responses.csv").write_text("a\n")
    # The inner loop pools, so the LAST experiment's CSV is the full training set.
    assert training_responses(run_root, 3).parent.parent.name == "experiment2"

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="No training responses"):
        training_responses(bare, 3)
    (bare / "pooled_responses.csv").write_text("a\n")
    assert training_responses(bare, 3).name == "pooled_responses.csv"


def test_final_step_reader_handles_missing_and_empty_csv(tmp_path):
    assert final_step_from_csv(tmp_path / "absent.csv") is None
    empty = tmp_path / "empty.csv"
    empty.write_text("rmse,best_model\n", encoding="utf-8")
    assert final_step_from_csv(empty) is None
    csv_path = tmp_path / "holdout.csv"
    csv_path.write_text(
        "rmse,best_model\n0.5,early\n0.1,final\n", encoding="utf-8"
    )
    assert final_step_from_csv(csv_path)["best_model"] == "final"


def test_summary_groups_by_ground_truth_and_averages():
    run = CeilingRun(
        cells=[
            CellCeiling("run1/a", "a", 10, 0.01, 0.0, 0.0, loop_rmse=0.05, gap_to_ceiling=0.04),
            CellCeiling("run2/a", "a", 10, 0.03, 0.0, 0.0, loop_rmse=0.09, gap_to_ceiling=0.06),
            CellCeiling("run1/b", "b", 10, 0.10, 0.0, 0.0, loop_rmse=None, gap_to_ceiling=None),
        ]
    )
    rows = {r["gt_model"]: r for r in summarize_by_ground_truth(run)}
    assert rows["a"]["n"] == 2
    assert rows["a"]["mean_ceiling_rmse"] == pytest.approx(0.02)
    assert rows["a"]["mean_gap"] == pytest.approx(0.05)
    assert rows["a"]["max_ceiling_rmse"] == pytest.approx(0.03)
    # A cell with no recorded final step contributes a ceiling but no gap.
    assert rows["b"]["mean_gap"] is None and rows["b"]["mean_loop_rmse"] is None


# --- the real thing -----------------------------------------------------------


@pytest.mark.slow
def test_ceiling_runs_end_to_end_on_a_real_archived_cell():
    """One real cell, tiny sampler, a few eval stimuli.

    Asserts the two things that have actually broken in this campaign's tooling:
    the archived run tree resolves (it is inside agent_runs.tar.gz, not on
    disk), and the evaluation path agrees with the target — ``oracle_rmse`` is
    the ground truth scored against itself and must be 0.
    """
    if not SWEEP_ROOT.is_dir():
        pytest.skip(f"no sweep at {SWEEP_ROOT}")
    cells = discover_cells(SWEEP_ROOT)
    cell_dir = cells[0].parent
    gt_dir = json.loads((cell_dir / "holdout.json").read_text())["seed_models_dir"]
    if not (Path(gt_dir) / f"{cell_dir.name}.py").is_file():
        pytest.skip(f"held-out registry gone: {gt_dir}")

    with CellArchiveManager() as manager:
        result = ceiling_for_cell(
            cell_dir,
            manager=manager,
            fit_kwargs={"draws": 50, "tune": 50, "chains": 2},
            predict_max_draws=20,
            limit_eval=40,
        )
    assert result.n_eval_stimuli == 40
    assert result.oracle_rmse == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= result.ceiling_rmse < 1.0
    assert result.gt_model == cell_dir.name
