# name: cell_response_consistency
# description: Mean over repeated (set,target) cells (2+ responses) of the within-cell Bernoulli variance p(1-p) of yes responses; observed < null means people agree with each other more (more deterministic answers) than the model's intermediate probabilities imply, observed > null means more disagreement.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = df["rating"].astype(float).groupby(key)
    p = g.mean(); c = g.size()
    p = p[c >= 2]
    return float((p * (1 - p)).mean()) if len(p) else 0.0
