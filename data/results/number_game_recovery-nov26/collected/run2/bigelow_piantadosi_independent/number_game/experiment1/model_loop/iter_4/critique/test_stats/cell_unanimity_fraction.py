# name: cell_unanimity_fraction
# description: Fraction of (set, target) cells answered by >1 response in which all responses agree (all yes or all no); t_observed above null_mean means people are more consistent per item (sharper, less graded endorsement) than the model's probabilities imply, below means less consistent.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] > 1]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
