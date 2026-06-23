"""Tests for the human-experiment seed-vs-best-agent fit comparison figure.

Each live outer-loop experiment leaves one ``model_posterior.json`` under::

    <runs_root>/run<r>/subjective_randomness/experiment<e>/model_loop/

It records every model's PSIS-LOO fit (``elpd_loo``) and an ``arviz.compare``
table (``comparison``) whose ``elpd_diff``/``dse`` are measured against the
overall best (rank-0) model. ``plot_human_fit_comparison`` turns these into a
forest array — one facet per (run, experiment) — comparing the four hand-written
seed models against the single best agent-proposed model, with the best agent at
0 and each seed at its ELPD-LOO difference ± its LOO standard error.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import plotnine
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "analysis"


def _load_cli():
    """Load the analysis script as a module (its helpers are the unit under test)."""
    spec = importlib.util.spec_from_file_location(
        "plot_human_fit_comparison", SCRIPTS / "plot_human_fit_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


# A single cell's posterior, hand-built so every plotted number is checkable.
# Best agent (overall best, rank 0) sits at elpd -100; the four seeds and one
# also-ran agent fall below it. ``elpd_diff``/``dse`` are relative to rank 0.
UNIT_CELL = {
    "run": "run1",
    "experiment": "experiment1",
    "elpd_loo": {
        "agent_winner": -100.0,
        "prototype_similarity": -110.0,
        "bayesian_diagnosticity": -115.0,
        "also_ran_agent": -125.0,
        "encoding_compressibility": -130.0,
        "window_typicality": -140.0,
    },
    "comparison": {
        "agent_winner": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0},
        "prototype_similarity": {"rank": 1, "elpd_diff": 10.0, "dse": 4.0},
        "bayesian_diagnosticity": {"rank": 2, "elpd_diff": 15.0, "dse": 5.0},
        "also_ran_agent": {"rank": 3, "elpd_diff": 25.0, "dse": 6.0},
        "encoding_compressibility": {"rank": 4, "elpd_diff": 30.0, "dse": 8.0},
        "window_typicality": {"rank": 5, "elpd_diff": 40.0, "dse": 9.0},
    },
}


def _posterior(best_agent, seeds, *, extra_agents=None):
    """Build a minimal ``model_posterior.json`` payload for an experiment cell.

    ``best_agent`` is ``(name, elpd)`` and must be the highest elpd. ``seeds`` and
    ``extra_agents`` map name -> elpd. ``elpd_diff``/``dse`` are derived so they
    are self-consistent with ``elpd_loo`` and measured against the best model.
    """
    elpd = {best_agent[0]: best_agent[1], **seeds, **(extra_agents or {})}
    top = best_agent[1]
    ranked = sorted(elpd.items(), key=lambda kv: kv[1], reverse=True)
    comparison = {
        name: {"rank": rank, "elpd_diff": top - e, "dse": float(rank)}
        for rank, (name, e) in enumerate(ranked)
    }
    return {"elpd_loo": elpd, "comparison": comparison, "n_trials": 100}


SEEDS_A = {
    "prototype_similarity": -110.0,
    "encoding_compressibility": -130.0,
    "bayesian_diagnosticity": -115.0,
    "window_typicality": -140.0,
}
# A later experiment's seeds: deeper (more negative) ELPD because it has more
# trials. The best agent must still beat every seed (it is rank 0 by construction).
SEEDS_B = {
    "prototype_similarity": -210.0,
    "encoding_compressibility": -218.0,
    "bayesian_diagnosticity": -212.0,
    "window_typicality": -250.0,
}


# --- unit: extracting plotted rows from one cell ---------------------------


def test_seed_vs_best_agent_rows_picks_best_agent_and_all_seeds():
    rows = cli.seed_vs_best_agent_rows(UNIT_CELL)
    by_model = {r["model"]: r for r in rows}
    # Exactly the four seeds plus the single best agent model — the also-ran is dropped.
    assert set(by_model) == {
        "agent_winner",
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
        "window_typicality",
    }
    assert by_model["agent_winner"]["kind"] == "best agent model"
    assert by_model["prototype_similarity"]["kind"] == "seed model"


def test_seed_vs_best_agent_rows_use_elpd_relative_to_best_agent():
    rows = {r["model"]: r for r in cli.seed_vs_best_agent_rows(UNIT_CELL)}
    # The best agent is the reference: at 0 with no error.
    assert rows["agent_winner"]["elpd_rel"] == pytest.approx(0.0)
    assert rows["agent_winner"]["dse"] == pytest.approx(0.0)
    # Seeds sit below it by -elpd_diff, with the dse carried straight through.
    assert rows["prototype_similarity"]["elpd_rel"] == pytest.approx(-10.0)
    assert rows["prototype_similarity"]["dse"] == pytest.approx(4.0)
    assert rows["window_typicality"]["elpd_rel"] == pytest.approx(-40.0)
    assert rows["window_typicality"]["dse"] == pytest.approx(9.0)
    # Error-bar bounds are mean ± dse.
    proto = rows["prototype_similarity"]
    assert proto["xmin"] == pytest.approx(-14.0)
    assert proto["xmax"] == pytest.approx(-6.0)
    # run/experiment labels ride along for faceting.
    assert proto["run"] == "run1" and proto["experiment"] == "experiment1"


def test_seed_vs_best_agent_rows_tolerate_rounded_elpd_loo():
    # The stored top-level elpd_loo is rounded (~4 dp) while comparison.elpd_diff is
    # full precision; disagreements within rounding must be accepted, not rejected.
    cell = {
        "run": "run1",
        "experiment": "experiment1",
        "elpd_loo": {**UNIT_CELL["elpd_loo"], "prototype_similarity": -110.0001},
        "comparison": {
            **UNIT_CELL["comparison"],
            "prototype_similarity": {"rank": 1, "elpd_diff": 10.00012345, "dse": 4.0},
        },
    }
    rows = {r["model"]: r for r in cli.seed_vs_best_agent_rows(cell)}
    assert rows["prototype_similarity"]["elpd_rel"] == pytest.approx(-10.00012345)


def test_seed_vs_best_agent_rows_reject_gross_elpd_mismatch():
    # A whole-unit disagreement between elpd_loo and the compare table is a real
    # inconsistency and must still fail loudly.
    cell = {
        "run": "run1",
        "experiment": "experiment1",
        "elpd_loo": {**UNIT_CELL["elpd_loo"], "prototype_similarity": -200.0},
        "comparison": UNIT_CELL["comparison"],
    }
    with pytest.raises(ValueError, match="disagrees"):
        cli.seed_vs_best_agent_rows(cell)


def test_seed_vs_best_agent_rows_fail_when_a_seed_is_missing():
    cell = {**UNIT_CELL, "elpd_loo": {
        k: v for k, v in UNIT_CELL["elpd_loo"].items() if k != "window_typicality"
    }}
    with pytest.raises(ValueError, match="window_typicality"):
        cli.seed_vs_best_agent_rows(cell)


def test_seed_vs_best_agent_rows_fail_when_best_agent_is_not_rank_zero():
    # Make a seed the overall best: the best agent is no longer rank 0, so the
    # dse-relative-to-best-agent error bars are unavailable and we must refuse.
    cell = {
        "run": "run1",
        "experiment": "experiment1",
        "elpd_loo": {**UNIT_CELL["elpd_loo"], "prototype_similarity": -90.0},
        "comparison": {
            "prototype_similarity": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0},
            "agent_winner": {"rank": 1, "elpd_diff": 10.0, "dse": 4.0},
            "bayesian_diagnosticity": {"rank": 2, "elpd_diff": 25.0, "dse": 5.0},
            "also_ran_agent": {"rank": 3, "elpd_diff": 35.0, "dse": 6.0},
            "encoding_compressibility": {"rank": 4, "elpd_diff": 40.0, "dse": 8.0},
            "window_typicality": {"rank": 5, "elpd_diff": 50.0, "dse": 9.0},
        },
    }
    with pytest.raises(ValueError, match="rank 0"):
        cli.seed_vs_best_agent_rows(cell)


# --- unit: locating and labelling cells on disk ----------------------------


def test_find_posterior_files_globs_run_experiment_tree(tmp_path):
    made = []
    for run in ("run1", "run2"):
        for exp in ("experiment1", "experiment2"):
            d = tmp_path / run / "subjective_randomness" / exp / "model_loop"
            d.mkdir(parents=True)
            f = d / "model_posterior.json"
            f.write_text("{}", encoding="utf-8")
            made.append(f)
    found = cli.find_posterior_files(tmp_path)
    assert found == sorted(made)


def test_cell_label_reads_run_and_experiment_from_path():
    p = Path(
        "data/results/human_experiment/run2/subjective_randomness/"
        "experiment3/model_loop/model_posterior.json"
    )
    assert cli.cell_label(p) == ("run2", "experiment3")


# --- unit: assembling the figure -------------------------------------------


def _two_by_two_cells():
    return [
        {"run": "run1", "experiment": "experiment1",
         **_posterior(("agent_a", -100.0), SEEDS_A)},
        {"run": "run1", "experiment": "experiment2",
         **_posterior(("inner_loop_model", -200.0), SEEDS_B)},
        {"run": "run2", "experiment": "experiment1",
         **_posterior(("agent_b", -105.0), SEEDS_A)},
        {"run": "run2", "experiment": "experiment2",
         **_posterior(("agent_c", -190.0), SEEDS_B)},
    ]


def test_human_fit_rows_cover_every_cell():
    rows = cli.human_fit_rows(_two_by_two_cells())
    cells = {(r["run"], r["experiment"]) for r in rows}
    assert cells == {
        ("run1", "experiment1"), ("run1", "experiment2"),
        ("run2", "experiment1"), ("run2", "experiment2"),
    }
    # Five plotted models per cell (four seeds + best agent) × four cells.
    assert len(rows) == 20


def test_prettify_label_spaces_underscores_and_run_numbers():
    assert cli.prettify_label("prototype_similarity") == "prototype similarity"
    assert cli.prettify_label("run1") == "run 1"
    assert cli.prettify_label("experiment3") == "experiment 3"
    assert cli.prettify_label("best agent model") == "best agent model"


def test_forest_frame_orders_models_with_best_agent_on_top():
    frame = cli.human_fit_forest_frame(cli.human_fit_rows(_two_by_two_cells()))
    order = list(frame["model_label"].cat.categories)
    # plotnine draws the last category at the top of the y-axis. Labels are
    # prettified for display (underscores -> spaces).
    assert order[-1] == "best agent model"
    assert set(order) == {
        "best agent model",
        "prototype similarity",
        "encoding compressibility",
        "bayesian diagnosticity",
        "window typicality",
    }


def test_forest_frame_prettifies_facet_labels():
    frame = cli.human_fit_forest_frame(cli.human_fit_rows(_two_by_two_cells()))
    assert set(frame["run"].cat.categories) == {"run 1", "run 2"}
    assert set(frame["experiment"].cat.categories) == {"experiment 1", "experiment 2"}


def test_forest_ggplot_returns_a_ggplot():
    frame = cli.human_fit_forest_frame(cli.human_fit_rows(_two_by_two_cells()))
    assert isinstance(cli.human_fit_forest_ggplot(frame), plotnine.ggplot)


def test_select_cells_filters_to_one_experiment():
    chosen = cli.select_cells(_two_by_two_cells(), "experiment2")
    assert {c["experiment"] for c in chosen} == {"experiment2"}
    assert len(chosen) == 2  # one per run


def test_select_cells_none_keeps_all():
    assert len(cli.select_cells(_two_by_two_cells(), None)) == 4


def test_select_cells_unknown_experiment_fails_loudly():
    with pytest.raises(ValueError, match="experiment9"):
        cli.select_cells(_two_by_two_cells(), "experiment9")


def test_forest_ggplot_handles_a_single_experiment():
    cells = cli.select_cells(_two_by_two_cells(), "experiment1")
    frame = cli.human_fit_forest_frame(cli.human_fit_rows(cells))
    assert isinstance(cli.human_fit_forest_ggplot(frame), plotnine.ggplot)


def test_plot_human_fit_forest_writes_a_figure(tmp_path):
    frame = cli.human_fit_forest_frame(cli.human_fit_rows(_two_by_two_cells()))
    out = tmp_path / "human_fit.pdf"
    cli.plot_human_fit_forest(frame, out)
    assert out.exists() and out.stat().st_size > 0


# --- integration: the CLI over a run tree ----------------------------------


def test_cli_builds_figure_and_csv_from_run_tree(tmp_path):
    runs_root = tmp_path / "human_experiment"
    for cell in _two_by_two_cells():
        d = (runs_root / cell["run"] / "subjective_randomness"
             / cell["experiment"] / "model_loop")
        d.mkdir(parents=True)
        payload = {"elpd_loo": cell["elpd_loo"], "comparison": cell["comparison"]}
        (d / "model_posterior.json").write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "figs"
    cli.main(cli.Args(runs_root=runs_root, out_dir=out_dir))

    assert (out_dir / "human_fit_comparison.pdf").exists()
    csv_path = out_dir / "human_fit_comparison.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "best agent model" in text and "prototype_similarity" in text


def test_cli_filters_to_one_experiment(tmp_path):
    runs_root = tmp_path / "human_experiment"
    for cell in _two_by_two_cells():
        d = (runs_root / cell["run"] / "subjective_randomness"
             / cell["experiment"] / "model_loop")
        d.mkdir(parents=True)
        payload = {"elpd_loo": cell["elpd_loo"], "comparison": cell["comparison"]}
        (d / "model_posterior.json").write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "figs"
    cli.main(cli.Args(runs_root=runs_root, out_dir=out_dir, experiment="experiment2"))

    # The filtered view gets its own filenames so it never clobbers the full grid.
    assert (out_dir / "human_fit_comparison_experiment2.pdf").exists()
    csv_path = out_dir / "human_fit_comparison_experiment2.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "experiment2" in text and "experiment1" not in text


def test_cli_fails_loudly_when_no_runs_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_posterior.json"):
        cli.main(cli.Args(runs_root=tmp_path / "empty", out_dir=tmp_path / "out"))
