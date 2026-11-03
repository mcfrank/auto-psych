# name: cell_unanimity_fraction
# description: Fraction of (set, target) cells with >=4 responses in which all responses agree (all yes or all no); observed > null means people's answers are more deterministic/consensual per item than the model's graded posterior probabilities imply (model too uncertain), observed < null means people are more heterogeneous.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = df["rating"].astype(float).groupby(key).agg(["mean", "size"])
    g = g[g["size"] >= 4]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
