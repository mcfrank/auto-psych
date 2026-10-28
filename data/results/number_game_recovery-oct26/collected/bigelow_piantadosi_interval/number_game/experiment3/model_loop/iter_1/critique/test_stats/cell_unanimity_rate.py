# name: cell_unanimity_rate
# description: Fraction of (number_set, target) cells with >=5 responses where all responses agree (all yes or all no); observed above null_mean means people respond more deterministically/consistently per stimulus than the model's graded p_yes predicts (model too noisy), below means the model is too deterministic.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["size", "mean"])
    g = g[g["size"] >= 5]
    if len(g) == 0:
        return 0.0
    m = g["mean"].to_numpy()
    return float(((m == 0) | (m == 1)).mean())
