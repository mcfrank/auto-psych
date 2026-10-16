# name: large_switch_gap_more_switches
# description: Among pairs of length >= 6 whose switch counts differ by at least 5, the rate of choosing the more-switching sequence; observed below null means people's choices saturate short of the near-certainty the model predicts for very lopsided pairs (more noise/reversal on easy pairs), above means people are even more decisive.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ka = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    kb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    n = a.str.len().to_numpy()
    m = (n >= 6) & (np.abs(ka - kb) >= 5)
    y = df["chose_left"].to_numpy()[m]
    ta = (ka > kb)[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
