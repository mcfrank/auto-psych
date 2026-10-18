# name: unanimous_cell_fraction
# description: Fraction of (set, target) cells with >=5 responses where all responses agree (all yes or all no); observed > null means human judgments are more extreme/deterministic per item than the model's graded probabilities predict (model too lukewarm), observed < null means the reverse.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 5]
    if len(g) == 0:
        return 0.0
    m = g["mean"].values
    return float(np.mean((m == 0) | (m == 1)))
