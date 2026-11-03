# name: cell_response_variance
# description: Mean within-cell Bernoulli variance p(1-p) of yes responses across repeated (set, target) cells with 5+ responses; observed above null means people disagree on the same item more than the model's predicted probabilities imply (model too extreme/confident), below means people are more unanimous than the model predicts.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = df["rating"].astype(float).groupby(key)
    cnt = g.size()
    p = g.mean()[cnt >= 5]
    return float((p * (1 - p)).mean()) if len(p) else 0.0
