# name: shared_tens_digit_out_of_span
# description: Among non-example targets outside the examples' span, yes-rate for targets sharing a tens digit (same decade, t//10) with some example minus targets sharing none; observed above null_mean means people generalize by decade/surface-digit similarity more than the model predicts.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    arr = {s: np.array([int(x) for x in s.split(",")]) for s in u}
    ns = df["number_set"].to_numpy()
    ex = np.full((len(df), 4), -1000)
    for s, a in arr.items():
        ex[ns == s, :len(a)] = a
    valid = ex > -1000
    lo = np.where(valid, ex, 1000).min(axis=1); hi = np.where(valid, ex, -1000).max(axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    dec = np.where(valid, ex // 10, -99)
    share = (dec == (t // 10)[:, None]).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    base = ((t < lo) | (t > hi)) & ~isex
    a = base & share; b = base & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
