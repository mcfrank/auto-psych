# name: within_pair_disagreement
# description: Mean unbiased Bernoulli variance n/(n-1)*p(1-p) of responses within identical (set, target) pairs answered >=2 times; observed < null means humans agree on a given item more than the model's predicted probabilities allow (model too uncertain/mushy), observed > null means more disagreement than predicted.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy(dtype=float); k = g["size"].to_numpy(dtype=float)
    return float(np.mean(p * (1 - p) * k / (k - 1)))
