"""The ablation figure (``scripts/paper/ablation_figure.py``): the backward
elimination of each domain's overall winner, drawn as the path of the model
kept after each round."""

from __future__ import annotations

from scripts.paper import ablation_figure


def _candidate(mechanism, elpd_cv, converged=True, se=4.0, elpd_se=30.0):
    return {"mechanism": mechanism, "elpd_cv": elpd_cv, "elpd_se": elpd_se, "converged": converged, "se": se}


ELIMINATION = {
    "rounds": [
        {"mechanisms": ["a", "b", "c"], "elpd_cv": -100.0, "elpd_se": 10.0, "converged": False,
         "candidates": [_candidate("a", -99.0), _candidate("b", -130.0)]},
        {"mechanisms": ["b", "c"], "elpd_cv": -99.0, "elpd_se": 11.0, "converged": True,
         # b is the cheapest in nats; c, unconverged, cannot be chosen.
         "candidates": [_candidate("b", -120.0, se=5.0, elpd_se=12.0), _candidate("c", -101.0, converged=False)]},
    ],
}


def test_the_path_is_each_rounds_model_relative_to_the_starting_model_with_its_elpds_se():
    path = ablation_figure.elimination_path(ELIMINATION)
    assert path["kept"] == [(3, 0.0, 10.0, False, None), (2, 1.0, 11.0, True, "a")]


def test_the_stop_is_the_last_rounds_best_converged_removal_with_its_elpds_se():
    assert ablation_figure.elimination_path(ELIMINATION)["stop"] == (1, -20.0, 12.0, "b")


def test_the_subjective_randomness_panel_is_the_final_that_won_on_the_pooled_data():
    figure_data = {"pooled": {"best_model": "w2", "winner_runs": {"w1": "run1", "w2": "run2"}}}
    final, path = ablation_figure.best_sr_elimination(figure_data)
    assert final == "w2" and path == ablation_figure.SR_ABLATION / "run2" / "elimination.json"


def test_a_panel_per_domain_in_the_papers_order_from_each_domains_ablation():
    figure_data = {"pooled": {"best_model": "w2", "winner_runs": {"w1": "run1", "w2": "run2"}}}
    sources = ablation_figure.elimination_sources(figure_data)
    assert [domain for domain, _ in sources] == ["subjective_randomness", "rsa", "number_game"]
    paths = dict(sources)
    assert paths["subjective_randomness"] == ablation_figure.SR_ABLATION / "run2" / "elimination.json"
    # Chain 0's final, the best on the pre-registered test (docs/auto_rsa/ABLATION_C0.md),
    # not the consolidation's superset, whose elimination could not start.
    assert paths["rsa"] == ablation_figure.here() / "data/rsa/c0_ablation/eliminate/elimination.json"
    assert paths["number_game"] == (ablation_figure.here() / "data/results/number_game_dataset-oct26"
                                    / "analysis/ablation/elimination.json")


def test_the_figure_is_drawn_from_the_committed_eliminations(tmp_path):
    ablation_figure.main(ablation_figure.Args(out_dir=tmp_path))
    assert (tmp_path / "fig_ablation.pdf").exists() and (tmp_path / "fig_ablation.png").exists()
