# name: within_set_response_variance
# description: Mean over sets (with 10+ trials) of the Bernoulli variance p(1-p) of yes responses within the set; observed > null means people's answers within a set are less deterministic/more mixed than the sharpened all-or-none model predicts, observed < null more deterministic.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy(dtype=float)
    return float(np.mean(p * (1 - p)))
