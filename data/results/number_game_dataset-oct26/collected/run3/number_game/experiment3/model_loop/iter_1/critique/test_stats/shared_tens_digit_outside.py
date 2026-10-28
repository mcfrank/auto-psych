# name: shared_tens_digit_outside
# description: Among targets outside the set's [min,max] range and not examples, yes-rate for targets sharing a tens digit (same decade, 100 counted as 10) with some example minus targets sharing none, restricted to targets within 10 of the nearest example; observed > null means people generalize by decade membership more than the model's decade-aligned interval bonus captures.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    out = (t < lo) | (t > hi)
    d = np.nanmin(np.abs(E - t[:, None]), axis=1)
    near = out & (d <= 10) & (d > 0)
    shared = np.any(np.floor(E / 10) == np.floor(t / 10)[:, None], axis=1)
    a = near & shared; b = near & ~shared
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
