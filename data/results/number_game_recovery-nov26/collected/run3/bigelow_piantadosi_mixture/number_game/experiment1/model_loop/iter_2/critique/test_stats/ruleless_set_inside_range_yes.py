# name: ruleless_set_inside_range_yes
# description: Yes-rate for non-example targets strictly inside the example range, for 2-4 example sets with no common rule (gcd 1, mixed parity, no shared last digit, no shared digit); observed above the null means people fill in the range of patternless sets more than the model (which can treat examples as noise), below means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    exi = ex.fillna(0).astype(int).values
    g = np.gcd.reduce(exi, axis=1)
    valid = ex.notna().values
    par = np.where(valid, exi % 2, -1)
    mixed = (par == 0).any(axis=1) & (par == 1).any(axis=1)
    last = np.where(valid, exi % 10, -1)
    shared_last = np.all((last == last[:, [0]]) | ~valid, axis=1)
    shared_digit = np.zeros(len(df), dtype=bool)
    for dgt in range(10):
        has = (exi % 10 == dgt) | ((exi // 10) % 10 == dgt) & (exi >= 10) | ((exi == 100) & (dgt in (0, 1)))
        shared_digit |= np.all(has | ~valid, axis=1)
    ruleless = (n.values >= 2) & (g == 1) & mixed & ~shared_last & ~shared_digit
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    is_ex = ex.eq(t, axis=0).any(axis=1)
    m = ruleless & (t > lo).values & (t < hi).values & ~is_ex.values
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
