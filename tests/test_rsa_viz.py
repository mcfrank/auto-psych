"""The figures' aggregate export (src.rsa.viz)."""

import json

import pandas as pd
import pytest

from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@pytest.mark.slow  # runs NUTS
def test_the_export_counts_choices_per_display_and_adds_each_models_predictions(tmp_path):
    from src.rsa.dataset import DEFAULT_TRIALS_CSV
    from src.rsa.viz import export_panels

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    pooled = tmp_path / "pooled.csv"
    df[df.experiment.isin(["E8_levels"])].assign(source="pragmods").to_csv(pooled, index=False)
    prereg = tmp_path / "prereg.csv"
    df[df.experiment.isin(["E9_twins"])].assign(source="auto_psych").to_csv(prereg, index=False)
    out = export_panels.main(export_panels.Args(
        pooled=pooled, prereg=prereg, cache=tmp_path / "cache", out=tmp_path / "panels.json",
        models=[f"l1={SEEDS / 'rsa_l1.py'}", f"literal={SEEDS / 'literal_listener.py'}"],
        num_warmup=100, num_samples=100, num_chains=2))
    datasets = {p["dataset"] for p in out["panels"]}
    assert datasets == {"literature", "prereg"}
    for p in out["panels"]:
        assert sum(c["count"] for c in p["classes"]) == p["n"]
        for label in ("l1", "literal"):
            assert sum(c["preds"][label] for c in p["classes"]) == pytest.approx(1.0, abs=1e-3)
    n_trials = len(df[df.experiment.isin(["E8_levels", "E9_twins"]) & df.included.astype(bool)
                      & (df.dv == "forced_choice") & (df.objects != "[]")])
    assert sum(p["n"] for p in out["panels"]) == n_trials
    text = (tmp_path / "panels.json").read_text()
    assert "participant_id" not in text and json.loads(text)["models"]["l1"]["model"] == "rsa_l1"


def test_figures_render_from_repo_data(tmp_path):
    """Every figure draws from the summaries in the repository (the scatter from the interim
    source when the Sherlock export is absent)."""
    from src.rsa.viz import figures

    paths = figures.main(figures.Args(out=tmp_path, panels=tmp_path / "absent.json"))
    assert [p.name for p in paths] == [
        "fig1_existing_data.png", "fig2_chain_trajectories.png", "fig3_prospective_tests.png",
        "fig4_prereg_all_models.png", "fig5_model_vs_data.png", "fig6_consolidation.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_interim_panels_hold_no_prereg_data_and_valid_probabilities():
    from src.rsa.viz.figures import interim_panels

    panels = interim_panels()["panels"]
    assert {p["dataset"] for p in panels} == {"literature", "campaign"}
    assert sum(p["dataset"] == "campaign" for p in panels) == 3 * 156
    for p in panels:
        assert sum(c["count"] for c in p["classes"]) == p["n"]
        for label in ("c0", "promoted", "starting"):
            assert abs(sum(c["preds"][label] for c in p["classes"]) - 1) < 1e-4


def test_the_full_scatter_and_its_per_dataset_version_come_from_the_export(tmp_path):
    from src.rsa.viz import figures

    panels = figures.here() / "data/rsa/viz/panels.json"
    figures.style()
    assert figures.fig_model_vs_data_by_dataset(tmp_path, panels).exists()
    data = json.loads(panels.read_text())
    assert {p["dataset"] for p in data["panels"]} == {"literature", "prereg", "campaign c0", "campaign c1",
                                                      "campaign c2"}
    assert sum(p["n"] for p in data["panels"] if p["dataset"] == "prereg") == 14404
    assert "participant" not in panels.read_text()
