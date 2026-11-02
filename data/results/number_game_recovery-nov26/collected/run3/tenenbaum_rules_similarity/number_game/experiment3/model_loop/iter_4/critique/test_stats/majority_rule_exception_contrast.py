# name: majority_rule_exception_contrast
# description: In sets of >=3 examples where all but exactly one example are multiples of some m in 3..12 (and not all are), yes rate for unshown multiples of m minus yes rate for unshown non-multiples; above the null means people keep the majority rule and discount the odd-one-out example more than the model (which only has power-of-2 exception rules and a uniform example-noise rate).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    sv = sets.values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    a_mask = np.zeros(len(df), bool)
    b_mask = np.zeros(len(df), bool)
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 3:
            continue
        ms = [m for m in range(3, 13) if sum(e % m == 0 for e in ex) == len(ex) - 1]
        if not ms:
            continue
        m = max(ms)
        r = sv == s
        unshown = ~np.isin(t, ex)
        a_mask |= r & unshown & (t % m == 0)
        b_mask |= r & unshown & (t % m != 0)
    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return 0.0
    return float(y[a_mask].mean() - y[b_mask].mean())
