# name: near_unanimous_pair_fraction
# description: Fraction of distinct pairs (side-independent, with at least 10 responses) on which at least 85% of responses picked the same sequence; observed above null_mean means people agree near-unanimously on more pairs than the model predicts (model too noisy/too much lapse on easy pairs), below means the model overstates consensus.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    cf = np.where(sw, cl, 1 - cl)
    g = pd.Series(cf).groupby(key).agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) == 0:
        return 0.0
    m = g["mean"].to_numpy()
    return float(np.mean(np.maximum(m, 1 - m) >= 0.85))
