# name: shared_last_digit_multi_boost
# description: Sets where all (>=2) examples share a last digit: yes-rate for outside-range targets with that digit minus other outside-range targets; observed vs null tells whether the model over- or under-produces noticed-digit rules.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    d = np.mod(np.nan_to_num(ex, nan=-1.0), 10)
    d0 = d[:, 0]
    same = (n >= 2) & np.all((d == d0[:, None]) | np.isnan(ex), axis=1)
    out = same & ((t < lo) | (t > hi))
    return rate(out & (np.mod(t, 10) == d0)) - rate(out & (np.mod(t, 10) != d0))
