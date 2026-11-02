# name: cell_unanimity_rate
# description: Fraction of distinct (number_set, target) cells (with >=5 responses) on which all responses agree (all yes or all no); observed above null means people agree more sharply per stimulus than the model's cell probabilities imply (model too graded/underconfident), below means too extreme.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 5]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
