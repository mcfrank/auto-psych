# name: in_range_width_slope
# description: Among multi-example sets, OLS slope of yes on example range width (max-min)/100 for non-member targets strictly inside [min,max]; observed more negative than null means people fill wide gaps less than the model's interval hypotheses do (less negative: fill them more).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    mem = np.any(ex == t[:, None], axis=1)
    m = (k >= 2) & (~mem) & (t > lo) & (t < hi)
    x = (hi - lo)[m] / 100.0
    y = df["rating"].to_numpy().astype(float)[m]
    if m.sum() < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
