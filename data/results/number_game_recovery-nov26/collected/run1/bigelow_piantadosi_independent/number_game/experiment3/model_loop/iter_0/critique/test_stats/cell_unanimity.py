# name: cell_unanimity
# description: Fraction of (set, target) cells whose responses are all identical (all yes or all no); observed above the null means people's judgments are more deterministic/consensual than the model's graded probabilities, below means more mixed.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
