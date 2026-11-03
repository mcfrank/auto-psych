# name: odd_set_odd_target_gap
# description: Among sets of 2+ examples that are all odd but not all prime, yes rate on odd non-example targets minus yes rate on even targets; observed above null means people generalise "odd numbers" more strongly than the model (whose odd cost suppresses it), below null means weaker.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    exi = np.where(valid, ex, 1).astype(int)
    allodd = np.where(valid, exi % 2 == 1, True).all(axis=1)
    primes = np.array([k > 1 and all(k % d for d in range(2, int(k ** 0.5) + 1)) for k in range(101)])
    allprime = np.where(valid, primes[np.clip(exi, 0, 100)], True).all(axis=1)
    t = df["target"].astype(int).to_numpy()
    isex = (ex == t[:, None]).any(axis=1)
    sel = (n >= 2) & allodd & ~allprime & ~isex
    r = df["rating"].astype(float).to_numpy()
    o = sel & (t % 2 == 1)
    e = sel & (t % 2 == 0)
    if o.sum() == 0 or e.sum() == 0:
        return 0.0
    return float(r[o].mean() - r[e].mean())
