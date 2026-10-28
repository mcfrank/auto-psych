"""The ablation figure (``scripts/paper/ablation_figure.py``): the backward
elimination of each domain's overall winner, drawn as the path of the model
kept after each round."""

from __future__ import annotations

from scripts.paper import ablation_figure


def _candidate(mechanism, elpd_cv, converged=True, se=4.0):
    return {"mechanism": mechanism, "elpd_cv": elpd_cv, "converged": converged, "se": se}


ELIMINATION = {
    "rounds": [
        {"mechanisms": ["a", "b", "c"], "elpd_cv": -100.0, "converged": False,
         "candidates": [_candidate("a", -99.0), _candidate("b", -130.0)]},
        {"mechanisms": ["b", "c"], "elpd_cv": -99.0, "converged": True,
         # b is the cheapest in nats; c, unconverged, cannot be chosen.
         "candidates": [_candidate("b", -120.0, se=5.0), _candidate("c", -101.0, converged=False)]},
    ],
}


def test_the_path_is_each_rounds_model_relative_to_the_starting_model():
    path = ablation_figure.elimination_path(ELIMINATION)
    assert path["kept"] == [(3, 0.0, False, None), (2, 1.0, True, "a")]


def test_the_stop_is_the_last_rounds_best_converged_removal():
    assert ablation_figure.elimination_path(ELIMINATION)["stop"] == (1, -20.0, "b")


def test_the_two_se_mark_is_the_bar_the_stopping_removal_fell_below():
    # 2 clustered SEs of the stopping comparison below the last kept model.
    assert ablation_figure.elimination_path(ELIMINATION)["two_se_mark"] == 1.0 - 2 * 5.0


def test_the_subjective_randomness_panel_is_the_final_that_won_on_the_pooled_data():
    figure_data = {"pooled": {"best_model": "w2", "winner_runs": {"w1": "run1", "w2": "run2"}}}
    final, path = ablation_figure.best_sr_elimination(figure_data)
    assert final == "w2" and path == ablation_figure.SR_ABLATION / "run2" / "elimination.json"


def test_the_figure_is_drawn_from_the_committed_eliminations(tmp_path):
    ablation_figure.main(ablation_figure.Args(out_dir=tmp_path))
    assert (tmp_path / "fig_ablation.pdf").exists() and (tmp_path / "fig_ablation.png").exists()
