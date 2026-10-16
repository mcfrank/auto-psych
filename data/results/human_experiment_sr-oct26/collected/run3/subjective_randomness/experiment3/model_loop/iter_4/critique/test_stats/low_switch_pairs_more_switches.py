# name: low_switch_pairs_more_switches
# description: Among pairs of length >= 6 where both sequences have at most 2 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed above null means people still reward the first breaks of a long streak more than the gambler's-run model predicts, below means the run-length term now over-rewards them.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ka = a.str.count("HT") + a.str.count("TH")
    kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = (n >= 6) & (ka <= 2) & (kb <= 2) & (ka != kb)
    y = df["chose_left"].to_numpy()[m.to_numpy()]
    ta = (ka > kb).to_numpy()[m.to_numpy()]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
