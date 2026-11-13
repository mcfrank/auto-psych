"""The ablation figure (``scripts/paper/ablation_figure.py``): the backward
elimination of each domain's overall winner, drawn as the path of the model
kept after each round."""

from __future__ import annotations

from scripts.paper import ablation_figure


def _candidate(mechanism, elpd_cv, converged=True, se=4.0, elpd_se=30.0, se_vs_start=6.0):
    return {"mechanism": mechanism, "elpd_cv": elpd_cv, "elpd_se": elpd_se, "converged": converged, "se": se,
            "se_vs_start": se_vs_start}


ELIMINATION = {
    "rounds": [
        {"mechanisms": ["a", "b", "c"], "elpd_cv": -100.0, "elpd_se": 10.0, "se_vs_start": 0.0, "converged": False,
         "candidates": [_candidate("a", -99.0), _candidate("b", -130.0)]},
        {"mechanisms": ["b", "c"], "elpd_cv": -99.0, "elpd_se": 11.0, "se_vs_start": 3.0, "converged": True,
         # b is the cheapest in nats; c, unconverged, cannot be chosen.
         "candidates": [_candidate("b", -120.0, se=5.0, elpd_se=12.0, se_vs_start=7.0),
                        _candidate("c", -101.0, converged=False)]},
    ],
}


def test_the_path_is_each_rounds_model_relative_to_the_starting_model_with_the_se_of_that_difference():
    # Each model's error bar is the paired SE of its ELPD-CV minus the full
    # model's: the y-axis's own difference (zero for the full model).
    path = ablation_figure.elimination_path(ELIMINATION)
    assert path["kept"] == [(3, 0.0, 0.0, False, None), (2, 1.0, 3.0, True, "a")]


def test_the_stop_is_the_last_rounds_best_converged_removal_with_the_se_of_its_difference():
    assert ablation_figure.elimination_path(ELIMINATION)["stop"] == (1, -20.0, 7.0, "b")


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
    # The November sweep's (from the 2016 paper's seeds), the definitive number-game runs.
    assert paths["number_game"] == (ablation_figure.here() / "data/results/number_game_dataset-nov26"
                                    / "analysis/ablation/elimination.json")


def test_a_missing_elimination_raises_unless_its_domain_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation_figure, "NG_ELIMINATION", tmp_path / "absent" / "elimination.json")
    import pytest

    with pytest.raises(FileNotFoundError, match="number_game.*--pending-domains"):
        ablation_figure.main(ablation_figure.Args(out_dir=tmp_path))
    ablation_figure.main(ablation_figure.Args(out_dir=tmp_path, pending_domains=["number_game"]))
    assert (tmp_path / "fig_ablation.pdf").exists() and (tmp_path / "fig_ablation.png").exists()


def test_the_figure_is_drawn_from_the_committed_eliminations(tmp_path):
    pending = [] if ablation_figure.NG_ELIMINATION.exists() else ["number_game"]
    ablation_figure.main(ablation_figure.Args(out_dir=tmp_path, pending_domains=pending))
    assert (tmp_path / "fig_ablation.pdf").exists() and (tmp_path / "fig_ablation.png").exists()


def _crowded_elimination(n=8):
    """A path whose steps fall fast and close together, like the number game's."""
    rounds = []
    mechanisms = [f"mechanism_{k}_cost" for k in range(n)]  # as long as the real names
    for k in range(n - 1):
        current = mechanisms[k:]
        rounds.append({"mechanisms": current, "elpd_cv": -100.0 - 3.0 * k * k, "elpd_se": 30.0,
                       "se_vs_start": 2.0 * k, "converged": True,
                       "candidates": [_candidate(current[0], -100.0 - 3.0 * (k + 1) ** 2, se_vs_start=2.0 * k + 2)]})
    return {"rounds": rounds}


def test_every_name_starts_at_its_own_point_and_no_two_names_overlap():
    # Unambiguous by construction: each name is anchored at its point (a flag
    # rising from it; the path's last point's name beside it), every point but
    # the full model's is named, and the names stay apart.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    path = ablation_figure.draw_panel(ax, _crowded_elimination(), "A. Crowded")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    names = ablation_figure.name_points(ax, path, renderer)
    steps = [(k[0], k[1]) for k in path["kept"]] + [path["stop"][:2]]
    assert sorted(i for _, i in names) == list(range(1, len(steps)))
    for name, i in names:
        assert name.xy == steps[i]
        dx, dy = name.xyann
        assert (dx * dx + dy * dy) ** 0.5 <= 8.0, (name.get_text(), name.xyann)  # points
    corners = [ablation_figure._text_corners(name, renderer) for name, _ in names]
    for a in range(len(corners)):
        for b in range(a + 1, len(corners)):
            assert not ablation_figure._polygons_overlap(corners[a], corners[b]), (names[a][0].get_text(),
                                                                                   names[b][0].get_text())
    plt.close(fig)


def test_the_number_game_panel_is_titled_number_sequence_generalization():
    assert ablation_figure.PANEL_TITLES["number_game"] == "Number sequence generalization"
