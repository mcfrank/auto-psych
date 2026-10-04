# name: short_length_more_switches
# description: Among pairs of length <= 4 whose switch counts differ, the rate of choosing the sequence with more switches; observed above/below null means the model under/over-states the switching preference on short sequences (its switching weight is not length-scaled).
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    n = a.str.len().to_numpy()
    sa = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    sb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    m = (n <= 4) & (sa != sb)
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(sa[m] > sb[m], c, 1 - c)))
