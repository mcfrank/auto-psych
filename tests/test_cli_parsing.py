"""CLI-layer tests: --experiments parsing and the argparse→tyro dataclasses.

The business logic each CLI calls is tested elsewhere; these pin the argument
*parsing* layer so the tyro conversion (arg names, Literal choices, multi-value
lists, optionality, defaults) cannot silently regress.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import tyro

from src.pipelines.outer_loop.orchestrator import experiment_dir, outer_data_dir
from src.pipelines.outer_loop.run import Args as OuterArgs
from src.pipelines.outer_loop.run import _parse_experiments
from src.pipelines.outer_loop.eig import Args as EigArgs
from src.pipelines.inner_loop.run import Args as InnerArgs
from src.model_comparison.likelihood import Args as LikelihoodArgs
from src.model_comparison.posterior import Args as PosteriorArgs

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── _parse_experiments ──────────────────────────────────────────────


def test_parse_experiments_single_number_expands_to_range():
    assert _parse_experiments("5") == [1, 2, 3, 4, 5]


def test_parse_experiments_one_is_singleton():
    assert _parse_experiments("1") == [1]


def test_parse_experiments_explicit_range_inclusive():
    assert _parse_experiments("4-6") == [4, 5, 6]


@pytest.mark.parametrize(
    "value,fragment",  # metachar-free substrings of the expected error message
    [
        ("0", "must be >= 1"),
        ("6-4", "range must have start <= end"),
        ("0-3", "range must have start <= end"),
        ("abc", "must be N "),
    ],
)
def test_parse_experiments_rejects_bad_input(value, fragment):
    with pytest.raises(ValueError) as excinfo:
        _parse_experiments(value)
    assert fragment in str(excinfo.value)


# ── tyro dataclass parsing ──────────────────────────────────────────


def test_posterior_cli_pools_multiple_responses():
    args = tyro.cli(
        PosteriorArgs,
        args=["--responses", "a.csv", "b.csv", "--models-dir", "m"],
    )
    assert args.responses == [Path("a.csv"), Path("b.csv")]
    assert args.models_dir == Path("m")
    assert args.out is None
    assert args.complexity_prior == 0.0


def test_likelihood_cli_required_fields_and_optional_cache():
    args = tyro.cli(
        LikelihoodArgs,
        args=["--responses", "r.csv", "--model", "alternation", "--models-dir", "m"],
    )
    assert args.responses == Path("r.csv")
    assert args.model == "alternation"
    assert args.cache_dir is None


def test_outer_run_cli_literal_choice_and_bool_flag():
    args = tyro.cli(
        OuterArgs,
        args=[
            "--project",
            "subjective_randomness",
            "--experiment",
            "1",
            "--agent",
            "2_design",
            "--validate",
        ],
    )
    assert args.project == "subjective_randomness"
    assert args.experiment == 1
    assert args.agent == "2_design"
    assert args.validate is True
    assert args.mode == "simulated_participants"  # default preserved


def test_eig_cli_defaults_and_required():
    args = tyro.cli(EigArgs, args=["--candidates", "c.json", "--models-dir", "m"])
    assert args.candidates == Path("c.json")
    assert args.top is None
    assert args.n_samples == 200


def test_inner_run_cli_required_paths():
    args = tyro.cli(
        InnerArgs,
        args=["--responses", "r.csv", "--seed-models", "s", "--results", "out"],
    )
    assert args.responses == Path("r.csv")
    assert args.seed_models == Path("s")
    assert args.results == Path("out")
    assert args.max_iterations == 0


# ── full-stack --help smoke (catches import-time errors + arg wiring) ─


@pytest.mark.parametrize(
    "module,expected_flag",
    [
        ("src.model_comparison.likelihood", "--model"),
        ("src.model_comparison.posterior", "--models-dir"),
        ("src.pipelines.outer_loop.run", "--project"),
        ("src.pipelines.outer_loop.eig", "--candidates"),
        ("src.pipelines.inner_loop.run", "--seed-models"),
    ],
)
def test_cli_module_help_runs(module, expected_flag):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert expected_flag in result.stdout


# ── output-path contract (the `Outputs:` line printed by outer run) ──


def test_experiment_outputs_resolve_under_data_outer_loop():
    # run.py prints f"Outputs: {outer_data_dir() / project_id}"; pin that target.
    project = "subjective_randomness"
    out_root = outer_data_dir() / project
    assert out_root == REPO_ROOT / "data" / "outer_loop" / project
    # experiment dirs live under that data root, not under src/.
    assert experiment_dir(project, 1) == out_root / "experiment1"
    assert "src" not in experiment_dir(project, 1).relative_to(REPO_ROOT).parts
