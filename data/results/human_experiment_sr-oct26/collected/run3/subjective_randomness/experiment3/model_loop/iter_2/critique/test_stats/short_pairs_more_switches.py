# name: short_pairs_more_switches
# description: Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed below null means people prefer alternation in very short sequences less than the model's length-scaled evidence predicts, above means more.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = (a.str.count("HT") + a.str.count("TH")).values
    sb = (b.str.count("HT") + b.str.count("TH")).values
    n = a.str.len().values
    y = df["chose_left"].values
    m = (n <= 3) & (sa != sb)
    if not m.any():
        return 0.5
    return float(np.where(sa > sb, y, 1 - y)[m].mean())
