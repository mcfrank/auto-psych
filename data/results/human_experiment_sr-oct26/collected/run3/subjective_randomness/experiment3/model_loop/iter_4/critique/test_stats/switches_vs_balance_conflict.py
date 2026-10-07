# name: switches_vs_balance_conflict
# description: Among length-8 pairs where the more-switching sequence has the more lopsided heads/tails count (|#H - #T| strictly larger), the rate of choosing the more-switching sequence; observed below null means people weigh heads/tails balance against switching more than the model predicts, above means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len().to_numpy()
    ia = np.abs(2 * a.str.count("H").to_numpy() - n)
    ib = np.abs(2 * b.str.count("H").to_numpy() - n)
    ka = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    kb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    m = (n == 8) & (((ka > kb) & (ia > ib)) | ((kb > ka) & (ib > ia)))
    y = df["chose_left"].to_numpy()[m]
    ta = (ka > kb)[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
