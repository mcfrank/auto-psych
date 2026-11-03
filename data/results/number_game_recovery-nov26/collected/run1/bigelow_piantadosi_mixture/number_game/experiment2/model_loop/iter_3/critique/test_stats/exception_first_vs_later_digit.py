# name: exception_first_vs_later_digit
# description: In 3-4 example sets where all but one example share a last digit d, yes-rate for non-example targets ending in d when the odd example is shown FIRST minus when it is shown later; observed differing from null_mean means the model's order dependence (sticky first guess) is too strong/weak — above means people are less hurt by an early exception than the model predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    best = np.zeros(len(t)); dig = np.full(len(t), -1.0)
    for k in range(10):
        c = np.sum(ld == k, axis=1)
        upd = c > best
        dig = np.where(upd, k, dig); best = np.where(upd, c, best)
    base = (n >= 3) & (best == n - 1) & ~isex & (np.mod(t, 10) == dig)
    first_odd = ld[:, 0] != dig
    a = base & first_odd; b = base & ~first_odd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
