# name: cell_response_agreement
# description: Mean within-cell pairwise agreement (fraction of response pairs that match) over (number_set, target) cells answered >=2 times, weighted by pairs; observed above null means people's answers to the same item are more deterministic/consistent than the model's middling p_yes predicts (model under-sharp), below means people are noisier.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["sum", "size"])
    g = g[g["size"] >= 2]
    k = g["sum"].to_numpy(float)
    m = g["size"].to_numpy(float)
    pairs = m * (m - 1) / 2
    agree = k * (k - 1) / 2 + (m - k) * (m - k - 1) / 2
    return float(agree.sum() / pairs.sum()) if pairs.sum() > 0 else 0.0
