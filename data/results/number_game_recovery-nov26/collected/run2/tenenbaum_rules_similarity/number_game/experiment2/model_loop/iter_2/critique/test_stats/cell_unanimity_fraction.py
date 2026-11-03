# name: cell_unanimity_fraction
# description: Fraction of (set, target) cells with at least 5 responses whose responses are all identical (all yes or all no); observed above the null means people respond more deterministically/consistently per stimulus than the model's intermediate posterior probabilities imply (below = people are more variable than the model).
def test_statistic(df):
    g = df.groupby([df["number_set"].astype(str), df["target"].astype(int)])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 5]
    if len(g) == 0:
        return 0.0
    m = g["mean"].to_numpy()
    return float(np.mean((m == 0) | (m == 1)))
