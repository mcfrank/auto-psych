"""Figure 3 row (c)'s search (scripts/paper/fig3_patterns.py): where the discovered model's advantage concentrates."""

import pandas as pd

from scripts.paper import fig3_patterns as P


def test_the_feature_value_carrying_the_advantage_ranks_first():
    # number-game items: the discovered model is right on single-example multiples, both models equal elsewhere
    rows = []
    for s in range(2, 12):
        for t, multiple in ((2 * s, True), (2 * s + 1, False)):
            rows.append({"number_set": str(s), "target": t, "n": 9, "observed": 0.9 if multiple else 0.4,
                         "best_model": 0.9 if multiple else 0.4, "best_seed": 0.4,
                         "best_model_lpd": -1.0 - 0.1 * s if multiple else -2.0, "best_seed_lpd": -2.0})
    r = P.rank(pd.DataFrame(rows), "number_game")
    assert (r.feature[0], r.value[0]) == ("after one example, the number is", "a multiple")
    assert r.z[0] > 0 and "n/a" not in set(r.value)


def test_outside_reference_games_the_advantage_is_the_held_out_log_density_difference():
    rows = [{"number_set": "2", "target": t, "n": 9, "observed": 0.5, "best_model": 0.5, "best_seed": 0.5,
             "best_model_lpd": -1.0 if t == 4 else -5.0, "best_seed_lpd": -5.0} for t in (3, 4)]
    assert list(P.advantage(pd.DataFrame(rows), "number_game")) == [0.0, 4.0]


def test_the_search_runs_on_figure_2s_held_out_predictions():
    import json
    from scripts.paper.figures import Args
    t = P.search_tables()
    fd = Args().figure_data
    for d in ("subjective_randomness", "number_game"):
        held = json.loads((fd / f"{d}_heldout_conditions.json").read_text())["conditions"]
        assert len(t[d]) == len(held) and "best_model_lpd" in t[d], d
    assert set(t["rsa"].dataset) == {"prereg"}
