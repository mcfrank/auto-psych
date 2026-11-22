"""Figure 3 row (c)'s search (scripts/paper/fig3_patterns.py): where the discovered model's advantage concentrates."""

import pandas as pd

from scripts.paper import fig3_patterns as P


def test_the_feature_value_carrying_the_advantage_ranks_first():
    # number-game items: the discovered model is right on single-example multiples, both models equal elsewhere
    rows = []
    for s in range(2, 12):
        for t, multiple in ((2 * s, True), (2 * s + 1, False)):
            rows.append({"number_set": str(s), "target": t, "n": 9, "observed": 0.9 if multiple else 0.4,
                         "best_model": 0.9 if multiple else 0.4, "best_seed": 0.4})
    r = P.rank(pd.DataFrame(rows), "number_game")
    assert (r.feature[0], r.value[0]) == ("after one example, the number is", "a multiple")
    assert r.z[0] > 0 and "n/a" not in set(r.value)
