# name: unanimity_rate
# description: Fraction of distinct (set, target, trial_index) items whose repeated responses are all identical (all yes or all no), among items answered 2+ times; observed > null means people agree more (model's predicted probabilities are too intermediate/uncertain), observed < null means the model is too deterministic.
def test_statistic(df):
    g = df.groupby(["number_set", "target", "trial_index"])["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 2]
    if len(g) == 0:
        return 0.0
    m = g["mean"].values
    return float(np.mean((m == 0) | (m == 1)))
