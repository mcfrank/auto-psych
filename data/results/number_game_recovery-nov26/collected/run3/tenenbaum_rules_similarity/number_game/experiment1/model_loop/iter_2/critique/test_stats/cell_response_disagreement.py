# name: cell_response_disagreement
# description: Mean over (set, target) cells (each answered by several participants) of the within-cell response variance p(1-p); observed below the model mean means people agree with each other more (respond more deterministically) than the model's graded yes-probabilities imply, observed above means more disagreement than the model's probabilities imply.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 2]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy(dtype=float)
    return float(np.mean(p * (1 - p)))
