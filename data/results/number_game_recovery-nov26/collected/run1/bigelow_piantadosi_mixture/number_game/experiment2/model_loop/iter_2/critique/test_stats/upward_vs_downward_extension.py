# name: upward_vs_downward_extension
# description: Multi-example sets: yes-rate for targets 1-10 above the max minus targets 1-10 below the min; nonzero observed vs null means an asymmetric extension the model misses.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    up = (n >= 2) & (t > hi) & (t <= hi + 10)
    dn = (n >= 2) & (t < lo) & (t >= lo - 10)
    return rate(up) - rate(dn)
