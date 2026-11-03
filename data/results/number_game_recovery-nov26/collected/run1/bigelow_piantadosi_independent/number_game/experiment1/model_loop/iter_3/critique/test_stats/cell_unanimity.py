# name: cell_unanimity
# description: Fraction of (set, target) cells with >=2 responses in which all responses agree; observed above null means responses are more deterministic/extreme (closer to 0 or 1 per item) than the model's graded posterior predicts, below null means more disagreement.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
