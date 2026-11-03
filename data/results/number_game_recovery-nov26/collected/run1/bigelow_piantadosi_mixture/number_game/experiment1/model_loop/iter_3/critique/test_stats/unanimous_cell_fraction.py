# name: unanimous_cell_fraction
# description: Fraction of (set, target) cells (each answered 5 times) where all answers agree; observed > null means human responses are more deterministic/consensual than the model's per-cell probabilities imply (model p too close to 0.5), observed < null means people are noisier than predicted.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
