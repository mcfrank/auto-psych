# name: pair_consensus_extremity
# description: Mean over unordered stimulus pairs of |P(choose the lexicographically smaller sequence) - 0.5|; t_observed > null_mean means people agree on pairs more strongly than the model's heterogeneity/noise allows, lower means the model is overconfident.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy(dtype=float)
    swap = a > b
    key = np.where(swap, b + "|" + a, a + "|" + b)
    chose_first = np.where(swap, 1 - y, y)
    p = pd.Series(chose_first).groupby(key).mean()
    return float(np.mean(np.abs(p.to_numpy() - 0.5)))
