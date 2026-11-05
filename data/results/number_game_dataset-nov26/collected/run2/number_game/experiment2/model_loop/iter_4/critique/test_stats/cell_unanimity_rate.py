# name: cell_unanimity_rate
# description: Fraction of (set, target) cells with >=3 answers where every answer agrees; observed > null means people are more consistent per item than the model's probabilities (predictions too close to 0.5), observed < null means the model is overconfident.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 3]
    if len(g) == 0:
        return 0.0
    m = g["mean"].to_numpy(float)
    return float(((m == 0) | (m == 1)).mean())
