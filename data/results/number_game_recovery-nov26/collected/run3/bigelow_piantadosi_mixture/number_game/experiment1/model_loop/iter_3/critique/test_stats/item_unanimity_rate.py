# name: item_unanimity_rate
# description: Fraction of (set, target) items (each answered by several participants) on which all answers agree; observed > null means people agree more (the model's item probabilities are too graded), observed < null means people disagree more than the model's item probabilities produce.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    m = g["mean"].values
    return float(np.mean((m == 0) | (m == 1)))
