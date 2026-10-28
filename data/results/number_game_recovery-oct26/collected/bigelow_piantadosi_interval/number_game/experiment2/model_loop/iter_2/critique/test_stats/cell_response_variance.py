# name: cell_response_variance
# description: Mean unbiased Bernoulli variance p(1-p)*n/(n-1) of responses within identical (set, target) cells asked >= 3 times; observed below the null means people agree on specific items more than the model's item-level predictions imply (the model is miscalibrated toward 0.5 on items people find clear-cut), above means people are less consistent than the model.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 3]
    if len(g) == 0:
        return 0.0
    v = g["mean"] * (1 - g["mean"]) * g["size"] / (g["size"] - 1)
    return float(np.average(v, weights=g["size"]))
