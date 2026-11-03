# name: within_cell_disagreement
# description: Mean within-(set,target) Bernoulli variance p(1-p) of responses across repeated presentations of the same cell; observed above null means people disagree more than the model's (too extreme) predicted probabilities imply, below null means the model is too graded/uncertain.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy()
    return float(np.mean(p * (1 - p)))
