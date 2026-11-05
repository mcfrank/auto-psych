# name: two_example_multiples_opposite_parity
# description: On 2-example same-parity sets whose examples are all multiples of some k in {3,4,6,7,8,9,11,12} (largest such k) and do not share a last digit, yes rate for non-example targets that are multiples of k but of the opposite parity (e.g. 28 after 7,63); observed above null means the model lets parity block arithmetic-rule extension too strongly (people extend the multiples rule across parity).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ez = np.where(valid, E, 0.0)
    lo = np.where(valid, E, np.inf).min(axis=1)
    hi = np.where(valid, E, -np.inf).max(axis=1)
    is_ex = (valid & (E == t[:, None])).any(axis=1)
    odd = (np.where(valid, Ez % 2, 1) == 1).all(axis=1)
    even = (np.where(valid, Ez % 2, 0) == 0).all(axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    k = np.zeros(len(t))
    for kk in [3, 4, 6, 7, 8, 9, 11, 12]:
        allm = (np.where(valid, Ez % kk, 0) == 0).all(axis=1)
        k = np.where(allm, kk, k)
    ld = np.where(valid, Ez % 10, np.nan)
    shared = (np.where(valid, ld == ld[:, [0]], True)).all(axis=1)
    kk_safe = np.where(k > 0, k, 1).astype(int)
    tm = (k > 0) & (t % kk_safe == 0)
    opp = np.where(odd, t % 2 == 0, np.where(even, t % 2 == 1, False))
    m = (n == 2) & (odd | even) & ~shared & ~is_ex & tm & opp
    return rate(m)
