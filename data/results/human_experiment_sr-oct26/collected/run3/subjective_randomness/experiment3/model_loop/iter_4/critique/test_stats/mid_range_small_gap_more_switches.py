# name: mid_range_small_gap_more_switches
# description: Among length-8 pairs where both sequences have 3-5 switches and their switch counts differ by 1 or 2, the rate of choosing the more-switching sequence; observed above null means people discriminate switch count in the middle of the range more sharply than the model predicts, below means less (a flatter preference near the 'ideal' switch rate).
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ka = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    kb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    n = a.str.len().to_numpy()
    d = np.abs(ka - kb)
    m = (n == 8) & (ka >= 3) & (ka <= 5) & (kb >= 3) & (kb <= 5) & (d >= 1) & (d <= 2)
    y = df["chose_left"].to_numpy()[m]
    ta = (ka > kb)[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
