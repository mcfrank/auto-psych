# name: alternation_pref_by_length_slope
# description: Slope over sequence length of the rate of choosing the sequence with more switches (pairs with unequal switch counts); observed above null_mean means the model under-predicts how the higher-switch preference grows with length.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = a.str.count("(?=HT|TH)").to_numpy(); sb = b.str.count("(?=HT|TH)").to_numpy()
    L = a.str.len().to_numpy().astype(float)
    m = sa != sb
    if m.sum() < 3:
        return 0.0
    y = df["chose_left"].to_numpy(dtype=float)[m]
    c = np.where((sa > sb)[m], y, 1 - y)
    x = L[m]
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, c, 1)[0])
