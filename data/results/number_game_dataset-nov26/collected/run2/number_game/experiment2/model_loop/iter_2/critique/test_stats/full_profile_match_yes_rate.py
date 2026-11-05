# name: full_profile_match_yes_rate
# description: Yes-rate on non-example targets (multi-example sets) that are inside [min,max] and satisfy every one of parity, multiple-of-3..10 and final digit that all examples share; observed below null means people cap their endorsement of best-fitting targets (a yes-ceiling/lapse the model's sigmoid lacks), above means the model is too lukewarm on perfect matches.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    valid = ~np.isnan(E)
    inset = np.any(E == t[:, None], axis=1)
    ok = np.ones(len(t), dtype=bool)
    # parity
    P = np.mod(E, 2)
    allodd = np.sum((P == 1) & valid, axis=1) == n
    alleven = np.sum((P == 0) & valid, axis=1) == n
    tp = np.mod(t, 2)
    ok &= ~(allodd & (tp != 1)) & ~(alleven & (tp != 0))
    for m in range(3, 11):
        allm = np.sum((np.mod(E, m) == 0) & valid, axis=1) == n
        ok &= ~(allm & (np.mod(t, m) != 0))
    D = np.mod(E, 10)
    dmin = np.nanmin(D, axis=1); dmax = np.nanmax(D, axis=1)
    samed = dmin == dmax
    ok &= ~(samed & (np.mod(t, 10) != dmin))
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = ok & (n >= 2) & (~inset) & (t >= lo) & (t <= hi)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
