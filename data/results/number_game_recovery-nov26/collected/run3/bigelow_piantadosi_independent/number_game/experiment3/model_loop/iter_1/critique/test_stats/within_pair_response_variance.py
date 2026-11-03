# name: within_pair_response_variance
# description: Mean over repeated (set, target) pairs of the within-pair Bernoulli variance p(1-p) of yes responses; observed above null means human answers to the same item are more mixed (less all-or-none) than the model's predicted probabilities, below null means humans are more unanimous than the model.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy(dtype=float)
    return float(np.mean(p * (1 - p)))
