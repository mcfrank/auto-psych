"""Collecting a finished holdout test-retest run into the repo must copy only the
lightweight result artifacts — the aggregate ``test_retest.{json,csv,png}`` and
the per-(repeat, ground-truth) ``holdout.{json,csv,png}`` — and render a compact
``SUMMARY.md`` from the aggregate JSON. The heavy material a run leaves on
scratch (per-task repo copies, MCMC ``.nc`` caches, the shared venv, agent XDG
state) must never be copied. Missing artifacts fail loudly.
"""

from __future__ import annotations

import json

import pytest

from src.subjective_randomness.results_collection import (
    collect_results,
    render_test_retest_summary,
)


def _summary_fixture() -> dict:
    """A minimal but schema-faithful test_retest.json payload (2 GTs x 2 runs)."""
    return {
        "runs_root": "/scratch/users/benpry/auto-psych/holdout_test_retest_full",
        "metric": "pearson_r",
        "n_runs_found": 2,
        "runs_found": ["run1", "run2"],
        "runs_missing_tidy": [],
        "gt_models": ["bayesian_diagnosticity", "window_typicality"],
        "runs_in_complete_matrix": ["run1", "run2"],
        "icc_2_1": 0.5812649309804439,
        "mean_pairwise_corr": 0.7042848132421727,
        "per_gt_model": {
            "bayesian_diagnosticity": {
                "n_runs": 2,
                "mean": 0.9573079865337373,
                "sd": 0.017968648722896254,
                "cv": 0.018769976826327257,
                "min": 0.9337813089436118,
                "max": 0.9768994211912475,
                "values": [0.976899, 0.943367],
                "modal_best_model": "iter1_candidate1",
                "best_model_agreement": 0.5,
            },
            "window_typicality": {
                "n_runs": 2,
                "mean": 0.9965026782084558,
                "sd": 0.004530981725738959,
                "cv": 0.004546883640980175,
                "min": 0.9907132650534649,
                "max": 0.9998949412883432,
                "values": [0.990713, 0.999778],
                "modal_best_model": "inner_loop_model",
                "best_model_agreement": 0.6,
            },
        },
    }


def _build_run_tree(root):
    """Write a fake finished test-retest run: aggregate + per-run artifacts plus
    the heavy material a real run leaves behind, which must NOT be collected."""
    summary = _summary_fixture()
    (root / "test_retest.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "test_retest.csv").write_text("gt_model,run,pearson_r\n", encoding="utf-8")
    (root / "test_retest.png").write_bytes(b"\x89PNG aggregate")

    for run in ("run1", "run2"):
        for gt in ("bayesian_diagnosticity", "window_typicality"):
            gt_dir = root / run / gt
            gt_dir.mkdir(parents=True)
            (gt_dir / "holdout.json").write_text('{"ok": true}', encoding="utf-8")
            (gt_dir / "holdout.csv").write_text("gt_model,step\n", encoding="utf-8")
            (gt_dir / "holdout.png").write_bytes(b"\x89PNG per-run")
            # Heavy material that must be excluded:
            (gt_dir / "mcmc_cache").mkdir()
            (gt_dir / "mcmc_cache" / "fit.nc").write_bytes(b"\x00" * 32)
            repo = gt_dir / "repo" / "src"
            repo.mkdir(parents=True)
            (repo / "leak.py").write_text("secret", encoding="utf-8")
    return summary


def test_collect_copies_lightweight_artifacts_and_writes_summary(tmp_path):
    source = tmp_path / "scratch_run"
    source.mkdir()
    _build_run_tree(source)
    dest = tmp_path / "repo" / "data" / "results" / "holdout_test_retest"

    report = collect_results(source, dest)

    # Aggregate copied.
    for ext in ("json", "csv", "png"):
        assert (dest / f"test_retest.{ext}").is_file()
    # Per-run artifacts copied, preserving run<r>/<gt>/ layout.
    assert (dest / "run1" / "bayesian_diagnosticity" / "holdout.json").is_file()
    assert (dest / "run2" / "window_typicality" / "holdout.png").is_file()
    # Generated summary.
    assert (dest / "SUMMARY.md").is_file()
    # Heavy material NOT copied.
    assert not list(dest.rglob("*.nc"))
    assert not list(dest.rglob("repo"))
    assert not list(dest.rglob("leak.py"))

    assert report.n_per_run_artifacts == 2 * 2 * 3  # 2 runs x 2 gts x {json,csv,png}


def test_summary_only_skips_per_run_artifacts(tmp_path):
    source = tmp_path / "scratch_run"
    source.mkdir()
    _build_run_tree(source)
    dest = tmp_path / "out"

    report = collect_results(source, dest, include_per_run=False)

    assert (dest / "test_retest.json").is_file()
    assert (dest / "SUMMARY.md").is_file()
    assert not (dest / "run1").exists()
    assert report.n_per_run_artifacts == 0


def test_collect_fails_loudly_when_aggregate_missing(tmp_path):
    source = tmp_path / "scratch_run"
    source.mkdir()  # no test_retest.* at all
    with pytest.raises(FileNotFoundError, match="test_retest"):
        collect_results(source, tmp_path / "out")


def test_collect_refuses_nonempty_destination_without_overwrite(tmp_path):
    source = tmp_path / "scratch_run"
    source.mkdir()
    _build_run_tree(source)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        collect_results(source, dest)


def test_render_summary_reports_reliability_and_per_gt_rows():
    md = render_test_retest_summary(_summary_fixture())
    assert "0.581" in md  # ICC(2,1)
    assert "0.704" in md  # mean pairwise correlation
    assert "bayesian_diagnosticity" in md
    assert "window_typicality" in md
    assert "iter1_candidate1" in md  # modal best model
    assert "0.957" in md  # per-gt mean, 3 d.p.


def test_render_summary_renders_none_as_na():
    summary = _summary_fixture()
    summary["icc_2_1"] = None
    summary["per_gt_model"]["window_typicality"]["best_model_agreement"] = None
    md = render_test_retest_summary(summary)
    assert "n/a" in md
