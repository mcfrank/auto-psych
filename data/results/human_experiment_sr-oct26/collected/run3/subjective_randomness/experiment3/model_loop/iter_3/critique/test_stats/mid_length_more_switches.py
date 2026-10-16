# name: mid_length_more_switches
# description: Among pairs of length 5-7 whose sequences differ in switch count, the rate of choosing the more-switching sequence; observed above null means people discriminate at intermediate lengths more decisively than the model's length scaling predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n >= 5) & (n <= 7) & (ka != kb)).to_numpy()
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    left_more = (ka > kb).to_numpy()[m]
    return float(np.mean(np.where(left_more, y, 1 - y)))
