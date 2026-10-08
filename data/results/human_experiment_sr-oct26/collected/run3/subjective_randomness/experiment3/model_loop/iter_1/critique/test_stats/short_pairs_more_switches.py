# name: short_pairs_more_switches
# description: Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed above null means people prefer alternation in very short sequences more than the model's length-scaled evidence predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = a.str.count("HT") + a.str.count("TH"); sb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n <= 3) & (sa != sb)).values
    if m.sum() == 0:
        return 0.5
    chose_more = np.where(sa.values > sb.values, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(chose_more[m].mean())
