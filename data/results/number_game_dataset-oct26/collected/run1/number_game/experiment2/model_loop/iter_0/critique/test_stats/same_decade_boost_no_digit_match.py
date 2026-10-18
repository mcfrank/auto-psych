# name: same_decade_boost_no_digit_match
# description: Among targets not sharing a final digit with any example (and not in the set), yes-rate for targets sharing a tens digit (decade) with some example minus others; observed above model mean means the model misses leading-digit/decade similarity.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    ten = np.floor(np.minimum(X, 99) / 10); tt = np.floor(np.minimum(t, 99) / 10)
    nodig = ~((X % 10) == (t[:, None] % 10)).any(axis=1)
    dec = (ten == tt[:, None]).any(axis=1)
    a = nodig & dec; b = nodig & ~dec
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
