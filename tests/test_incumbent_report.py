"""Tests for scripts/subjective_randomness/incumbent_report.py: the incumbent
record over a finished holdout sweep, read from each cell's archived
``agent_runs.tar.gz`` or its kept repo copy — the reporting that validates the
plan's baseline (0 incumbent changes over 27 steps in the archived
``motif_stack`` cells) and that the results phase reads."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from src.subjective_randomness.incumbent import cell_histories


def _entry(step, iteration, best, posteriors=None):
    return {
        "step": step,
        "iteration": iteration,
        "best_model": best,
        "posteriors": posteriors if posteriors is not None else {best: 1.0},
        "elpd_loo": {},
    }


def _seed_step(best, starting=("seed_a", "seed_b")):
    return _entry(0, None, best, posteriors={name: 1.0 / len(starting) for name in starting})


def _archived_cell(cell_dir: Path, gt: str, histories) -> None:
    _finished(cell_dir)
    staging = cell_dir / "_staging"
    for exp_num, history in enumerate(histories, start=1):
        loop_dir = staging / "_runs" / gt / f"experiment{exp_num}" / "model_loop"
        loop_dir.mkdir(parents=True)
        (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    with tarfile.open(cell_dir / "agent_runs.tar.gz", "w:gz") as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(staging)))
    import shutil

    shutil.rmtree(staging)


def _finished(cell_dir: Path) -> None:
    """The cell's result: only a finished cell is reported."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "holdout.json").write_text("{}", encoding="utf-8")


def _live_cell(cell_dir: Path, gt: str, histories) -> None:
    _finished(cell_dir)
    for exp_num, history in enumerate(histories, start=1):
        loop_dir = cell_dir / "repo" / "_runs" / gt / f"experiment{exp_num}" / "model_loop"
        loop_dir.mkdir(parents=True)
        (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")


def _build_sweep(root: Path) -> Path:
    # run1/alpha: archived, the incumbent never moves (the baseline shape).
    _archived_cell(root / "run1" / "alpha", "alpha", [
        [_seed_step("seed_a"), _entry(1, 0, "seed_a")],
        [_seed_step("seed_a"), _entry(1, 0, "seed_a")],
    ])
    # run2/alpha: archived, a seed rival takes over inside experiment 1.
    _archived_cell(root / "run2" / "alpha", "alpha", [
        [_seed_step("seed_a"), _entry(1, 0, "seed_b")],
        [_seed_step("seed_b")],
    ])
    # run1/beta: kept repo copy, a discovered model wins experiment 2.
    _live_cell(root / "run1" / "beta", "beta", [
        [_seed_step("seed_a"), _entry(1, 0, "seed_a")],
        [_seed_step("seed_a"), _entry(1, 0, "new_model")],
    ])
    return root


def test_cell_histories_reads_an_archive_or_a_kept_copy_and_refuses_neither(tmp_path):
    sweep = _build_sweep(tmp_path / "sweep")
    assert len(cell_histories(sweep / "run1" / "alpha")) == 2
    assert cell_histories(sweep / "run1" / "beta")[1][1]["best_model"] == "new_model"
    empty = tmp_path / "sweep" / "run3" / "alpha"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="agent_runs.tar.gz"):
        cell_histories(empty)


def test_report_records_every_cell_and_the_totals(tmp_path, capsys):
    from scripts.subjective_randomness.incumbent_report import Args, main

    sweep = _build_sweep(tmp_path / "sweep")
    out_md = tmp_path / "incumbent.md"
    main(Args(sweep=sweep, out=out_md))

    report = json.loads(out_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert list(report["cells"]) == ["run1/alpha", "run1/beta", "run2/alpha"]
    assert report["cells"]["run1/alpha"] == {
        "starting_models": ["seed_a", "seed_b"],
        "n_steps": 4,
        "n_incumbent_changes": 0,
        "n_steps_discovered_incumbent": 0,
        "final_incumbent": "seed_a",
        "changes": [],
    }
    assert report["cells"]["run2/alpha"]["n_incumbent_changes"] == 1
    assert report["cells"]["run2/alpha"]["n_steps_discovered_incumbent"] == 0
    assert report["cells"]["run1/beta"]["n_incumbent_changes"] == 1
    assert report["cells"]["run1/beta"]["n_steps_discovered_incumbent"] == 1
    assert report["cells"]["run1/beta"]["changes"] == [
        {"global_step": 3, "experiment": 2, "step": 1, "from": "seed_a", "to": "new_model"}
    ]
    assert report["totals"] == {
        "n_cells": 3,
        "n_steps": 11,
        "n_incumbent_changes": 2,
        "n_steps_discovered_incumbent": 1,
        "n_cells_with_a_change": 2,
        "n_cells_with_a_discovered_incumbent": 1,
    }

    md = out_md.read_text(encoding="utf-8")
    assert "| run1/alpha | 4 | 0 | 0 | seed_a |" in md
    assert "seed_a -> new_model (experiment 2, step 1)" in md
    assert "3 cells, 11 steps, 2 incumbent changes" in md
    # The table is also printed, so a Slurm log carries the numbers.
    assert "run2/alpha" in capsys.readouterr().out


def test_report_can_be_restricted_to_one_ground_truth(tmp_path):
    from scripts.subjective_randomness.incumbent_report import Args, main

    sweep = _build_sweep(tmp_path / "sweep")
    out_md = tmp_path / "alpha.md"
    main(Args(sweep=sweep, out=out_md, gt_model="alpha"))
    report = json.loads(out_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert list(report["cells"]) == ["run1/alpha", "run2/alpha"]
    assert report["totals"]["n_cells"] == 2


def test_report_fails_loudly_on_an_empty_selection_or_an_unreadable_cell(tmp_path):
    from scripts.subjective_randomness.incumbent_report import Args, main

    sweep = _build_sweep(tmp_path / "sweep")
    with pytest.raises(FileNotFoundError, match="gamma"):
        main(Args(sweep=sweep, out=tmp_path / "none.md", gt_model="gamma"))
    _finished(sweep / "run3" / "beta")  # finished, but its run record is gone
    with pytest.raises(FileNotFoundError, match="run3/beta"):
        main(Args(sweep=sweep, out=tmp_path / "broken.md"))
    with pytest.raises(FileNotFoundError, match="sweep root"):
        main(Args(sweep=tmp_path / "missing", out=tmp_path / "missing.md"))
