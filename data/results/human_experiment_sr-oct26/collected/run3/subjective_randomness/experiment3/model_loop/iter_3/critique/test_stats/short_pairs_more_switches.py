# name: short_pairs_more_switches
# description: Among pairs of length 2-3 whose sequences differ in switch count, the rate of choosing the sequence with more switches; observed below null means people prefer alternation in very short pairs less than the pseudo-flip-diluted model predicts, above means more.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = (n <= 3) & (ka != kb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m.to_numpy()]
    left_more = (ka > kb).to_numpy()[m.to_numpy()]
    return float(np.mean(np.where(left_more, y, 1 - y)))
