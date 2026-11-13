"""``scripts/paper/ng_finals_comparison.py`` compares the finals the runs actually exported."""

import json

from scripts.paper import ng_finals_comparison as fc


def test_the_finals_are_each_runs_exported_winner():
    for run, name in fc.FINALS.items():
        loop = fc.COLLECTED / f"run{run}" / "number_game" / "experiment3"
        assert json.loads((loop / "model_loop" / "model_posterior.json").read_text())["best_model"] == name
        assert (loop / "cognitive_models" / f"{name}.py").exists()
