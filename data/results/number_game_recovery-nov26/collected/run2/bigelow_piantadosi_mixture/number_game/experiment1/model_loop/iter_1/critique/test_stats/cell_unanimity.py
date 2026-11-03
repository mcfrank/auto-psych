# name: cell_unanimity
# description: Fraction of (number_set, target) cells with >=2 responses where all responses agree; observed above null means people are more consistent/extreme than the model's graded p_yes (model probabilities too intermediate), below means people are noisier.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
