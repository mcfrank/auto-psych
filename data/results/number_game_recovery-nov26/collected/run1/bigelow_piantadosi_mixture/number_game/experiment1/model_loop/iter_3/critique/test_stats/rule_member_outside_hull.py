# name: rule_member_outside_hull
# description: Among multi-example sets fitting a sparse rule (<=34 members), yes-rate for rule-member targets lying outside the examples' range minus rule-member targets inside the range; observed > null (closer to 0) means people extend rules beyond the examples' range more than the model (which lets range/interval guesses suppress them), observed < null means less.
def test_statistic(df):

    N = np.arange(1, 101)
    rules = [N % 2 == 1, N % 2 == 0, np.isin(N, [n * n for n in range(1, 11)]), np.isin(N, [1, 8, 27, 64])]
    rules += [N % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    rules += [np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]) for m in range(2, 11)]
    rules += [N % 10 == d for d in range(10)]
    R = np.array(rules)  # (rules, 100)
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.isfinite(ex).sum(axis=1)
    exi = np.where(np.isfinite(ex), ex, 1).astype(int) - 1
    ok = np.ones((len(df), R.shape[0]), bool)
    for j in range(ex.shape[1]):
        ok &= R[:, exi[:, j]].T | ~np.isfinite(ex[:, j])[:, None]
    sizes = np.where(ok, R.sum(axis=1)[None, :], 10 ** 6)
    best = sizes.argmin(axis=1); has = (sizes.min(axis=1) <= 34) & (n >= 2)
    t = df["target"].astype(int).to_numpy(); r = df["rating"].astype(float).to_numpy()
    inrule = R[best, t - 1]
    isex = (ex == t[:, None]).any(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    inhull = (t >= lo) & (t <= hi)
    a = has & ~isex & inrule & ~inhull; b = has & ~isex & inrule & inhull
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
