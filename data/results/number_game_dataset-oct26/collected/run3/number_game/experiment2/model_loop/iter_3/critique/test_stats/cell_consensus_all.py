# name: cell_consensus_all
# description: Over all (set, target) cells with at least 5 responses, the mean of p*(1-p) where p is the cell's yes-rate; observed < model means people's answers to the same item are more consensual / all-or-none than the model's graded probabilities produce, observed > model means more within-item disagreement than predicted.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 5]
    if len(g) == 0:
        return 0.0
    p = g["mean"].values
    return float(np.mean(p * (1 - p)))
