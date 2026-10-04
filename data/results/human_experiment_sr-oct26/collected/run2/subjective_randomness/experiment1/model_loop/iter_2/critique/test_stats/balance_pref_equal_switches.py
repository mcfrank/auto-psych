# name: balance_pref_equal_switches
# description: Among pairs with equal switch counts but unequal |#H-#T|, the rate of choosing the more H/T-balanced sequence; observed above null_mean means the model under-weights balance.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = a.str.count("(?=HT|TH)").to_numpy(); sb = b.str.count("(?=HT|TH)").to_numpy()
    la = a.str.len().to_numpy(); lb = b.str.len().to_numpy()
    ia = np.abs(2 * a.str.count("H").to_numpy() - la); ib = np.abs(2 * b.str.count("H").to_numpy() - lb)
    m = (sa == sb) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    left_bal = (ia < ib)[m]
    return float(np.mean(np.where(left_bal, y, 1 - y)))
