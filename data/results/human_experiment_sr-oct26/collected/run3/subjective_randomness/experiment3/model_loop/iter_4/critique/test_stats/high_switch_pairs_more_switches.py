# name: high_switch_pairs_more_switches
# description: Among length-8 pairs where both sequences have >= 5 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed below null means people penalise over-alternation near the top of the switch range more than the model predicts, above means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ka = a.str.count("HT") + a.str.count("TH")
    kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n == 8) & (ka >= 5) & (kb >= 5) & (ka != kb)).to_numpy()
    y = df["chose_left"].to_numpy()[m]
    ta = (ka > kb).to_numpy()[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
