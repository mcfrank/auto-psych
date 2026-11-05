# name: majority_digit_containment_lift
# description: In sets of 3-4 examples, yes-rate on non-example targets containing a digit that appears in at least half but not all of the examples, minus yes-rate on targets containing no digit from any example; the model only credits digits shared by ALL examples (plus the last example's final digit), so observed above null means people use partially-shared (majority) digit features the model misses.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    valid = ~np.isnan(E)
    Ei = np.where(valid, E, -1).astype(int)
    inset = np.any(E == t[:, None], axis=1)
    def has(x, d):
        # x int array (may be -1 for missing); digit containment for 1..100
        ones = np.mod(x, 10); tens = np.mod(x // 10, 10)
        h = (ones == d) | ((x >= 10) & (tens == d))
        if d in (0, 1):
            h = h | (x == 100)
        return h & (x >= 1)
    maj = np.zeros(len(t), dtype=bool)
    anyc = np.zeros(len(t), dtype=bool)
    for d in range(10):
        cnt = np.sum(has(Ei, d), axis=1)
        th = has(t, d)
        maj |= th & (cnt * 2 >= n) & (cnt < n)
        anyc |= th & (cnt >= 1)
    ok = (n >= 3) & (~inset)
    a = ok & maj
    b = ok & (~anyc)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
