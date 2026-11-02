# name: shared_last_digit_match_yes_rate
# description: Among sets of 2+ examples all ending in the same digit, yes rate for unseen targets that also end in that digit; observed < null means the model over-commits to "ends in d" rules (people endorse them less), observed > null under-commits.
def test_statistic(df):
    a = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    valid = ~np.isnan(a)
    n = valid.sum(axis=1)
    d = np.where(valid, np.mod(a, 10), np.nan)
    d0 = d[:, 0]
    same = (np.where(valid, d == d0[:, None], True)).all(axis=1)
    inset = (a == t[:, None]).any(axis=1)
    m = (n >= 2) & same & ~inset & (np.mod(t, 10) == d0)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
