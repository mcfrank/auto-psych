# name: alternation_pref_by_length_slope
# description: OLS slope over sequence length of the rate of choosing the sequence with more switches (pairs with unequal switch counts); observed above null_mean means the model under-predicts how the higher-switch preference grows with length (below: over-predicts), i.e. the length scaling is still wrong.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    sw = {s: sum(1 for x, y in zip(s, s[1:]) if x != y) for s in u}
    sa = a.map(sw).to_numpy(); sb = b.map(sw).to_numpy()
    m = sa != sb
    L = a.str.len().to_numpy(dtype=float)[m]
    y = df["chose_left"].to_numpy(dtype=float)[m]
    pick = np.where((sa > sb)[m], y, 1 - y)
    if L.size < 2 or np.var(L) == 0:
        return 0.0
    return float(np.cov(L, pick, bias=True)[0, 1] / np.var(L))
