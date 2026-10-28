# name: cell_unanimity_rate
# description: Fraction of (number_set, target) cells answered more than once in which every response is identical; observed above null means people agree more deterministically than the model predicts (its guessing/posterior is too noisy, cell probabilities not extreme enough), below means less.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    m = g["mean"].to_numpy()
    return float(np.mean((m == 0) | (m == 1)))
