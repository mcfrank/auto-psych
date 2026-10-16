# name: high_switch_pairs_more_switches
# description: Among length-8 pairs where both sequences have >= 5 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed below null means people penalise over-alternation near the top of the switch range more than the model predicts, above means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n == 8) & (ka >= 5) & (kb >= 5) & (ka != kb)).to_numpy()
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    left_more = (ka > kb).to_numpy()[m]
    return float(np.mean(np.where(left_more, y, 1 - y)))
