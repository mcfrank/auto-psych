# name: low_switch_pairs_more_switches
# description: Among pairs of length >= 6 where both sequences have at most 2 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed above null means people reward the first one or two breaks of a streak more than the model predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n >= 6) & (ka <= 2) & (kb <= 2) & (ka != kb)).to_numpy()
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    left_more = (ka > kb).to_numpy()[m]
    return float(np.mean(np.where(left_more, y, 1 - y)))
