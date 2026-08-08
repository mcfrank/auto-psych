"""Tests for the EIG scaling benchmark (scripts/analysis/benchmark_eig_scaling.py).

The benchmark answers: how many candidate stimuli can we afford to score with
EIG at design time? It times the pipeline's per-stimulus EIG path
(``src.pipelines.outer_loop.eig.annotate``) and a batched prior-predictive path
(one ``sample_prior_predictive`` per model for *all* candidates) on the same
models, checks the two paths agree on EIG values, and projects the maximum
candidate count per time budget from a linear fit of the timing curves.

The integration test runs the whole benchmark against the tiny fixture models
(prior-predictive only, no NUTS) — marked slow to be safe, consistent with
test_eig_pymc.py.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_MODELS = REPO_ROOT / "tests" / "fixtures" / "pymc_models"
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "benchmark_eig_scaling.py"


def _load_benchmark():
    """Load the benchmark script as a module (its helpers are under test)."""
    import sys

    spec = importlib.util.spec_from_file_location("benchmark_eig_scaling", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pair_universe_size_matches_enumeration():
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs

    mod = _load_benchmark()
    assert mod.pair_universe_size((4, 5)) == len(enumerate_all_pairs([4, 5]))
    # Duplicate lengths are ignored, matching enumerate_all_pairs semantics.
    assert mod.pair_universe_size((4, 4, 5)) == mod.pair_universe_size((4, 5))
    # Lengths 4-8: 496 sequences -> C(496, 2) pairs.
    assert mod.pair_universe_size((4, 5, 6, 7, 8)) == math.comb(496, 2)
    with pytest.raises(ValueError):
        mod.pair_universe_size(())


def test_fit_linear_seconds_recovers_exact_line():
    mod = _load_benchmark()
    sizes = [10, 20, 40]
    seconds = [1.0 + 0.5 * n for n in sizes]
    intercept, slope = mod.fit_linear_seconds(sizes, seconds)
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mod.fit_linear_seconds([10], [6.0])  # one point cannot pin down a line


def test_effective_time_model_uses_fit_when_slope_positive():
    mod = _load_benchmark()
    intercept, slope, resolved = mod.effective_time_model([10, 20, 40], [6.0, 11.0, 21.0])
    assert resolved is True
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(0.5)


def test_effective_time_model_falls_back_when_slope_in_noise():
    # Flat (or decreasing) timings: the marginal cost is below measurement
    # noise, so extrapolate conservatively from throughput at the largest size
    # (folding all overhead into the per-stimulus cost) instead of failing.
    mod = _load_benchmark()
    intercept, slope, resolved = mod.effective_time_model([4, 8], [0.05, 0.048])
    assert resolved is False
    assert intercept == 0.0
    assert slope == pytest.approx(0.048 / 8)


def test_max_candidates_within_budget():
    mod = _load_benchmark()
    # 1s overhead + 0.5s per candidate: an 11s budget fits 20 candidates.
    assert mod.max_candidates_within(11.0, intercept=1.0, slope=0.5) == 20
    # Budget below the fixed overhead fits none.
    assert mod.max_candidates_within(0.5, intercept=1.0, slope=0.5) == 0
    with pytest.raises(ValueError):
        mod.max_candidates_within(10.0, intercept=1.0, slope=0.0)


@pytest.mark.slow
def test_benchmark_end_to_end(tmp_path):
    """The full benchmark runs on the fixture models and answers the question."""
    mod = _load_benchmark()

    result = mod.run_benchmark(
        models_dir=FIXTURE_MODELS,
        per_stim_sizes=(2, 3),
        batched_sizes=(4, 8),
        n_samples=25,
        seed=7,
        lengths=(4, 5),
        budgets_s=(60.0, 600.0),
        agreement_n=4,
        out_dir=tmp_path,
    )

    # Timing records: both methods, all requested sizes, positive wall time.
    by_method = {}
    for rec in result.records:
        by_method.setdefault(rec.method, set()).add(rec.n_stimuli)
        assert rec.seconds > 0
        assert rec.seconds_per_stimulus == pytest.approx(rec.seconds / rec.n_stimuli)
        assert rec.n_models == 2
        assert rec.n_samples == 25
    assert by_method == {"per_stimulus": {2, 3}, "batched": {4, 8}}

    # The batched path must reproduce the pipeline path's EIG values.
    assert result.max_abs_eig_diff < 1e-6

    # Universe over lengths (4, 5): 2**4 + 2**5 = 48 sequences -> C(48, 2) pairs.
    assert result.universe_size == math.comb(48, 2)

    # Budget projections: positive, non-decreasing in budget, for both methods.
    for method in ("per_stimulus", "batched"):
        ns = [result.projections[method][b] for b in (60.0, 600.0)]
        assert all(n > 0 for n in ns)
        assert ns[0] <= ns[1]

    # Artifacts on disk: timings CSV (one row per record) + JSON summary + plot.
    csv_path = tmp_path / "eig_scaling_timings.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.records)

    summary = json.loads((tmp_path / "eig_scaling_summary.json").read_text())
    assert summary["universe_size"] == result.universe_size
    assert summary["max_abs_eig_diff"] == pytest.approx(result.max_abs_eig_diff)
    assert (tmp_path / "eig_scaling.png").exists()
