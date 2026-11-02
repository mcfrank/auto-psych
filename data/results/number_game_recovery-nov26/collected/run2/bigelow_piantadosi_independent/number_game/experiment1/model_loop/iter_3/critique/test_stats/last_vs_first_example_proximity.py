# name: last_vs_first_example_proximity
# description: Yes rate for non-example targets within 5 of the LAST-shown example but >5 from the first-shown, minus the rate for targets within 5 of the FIRST-shown but >5 from the last (multi-example sets); the model is order-invariant, so t_observed above null_mean means recency (last example weighted more), below means primacy.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].to_numpy(dtype=float)
    sv = np.where(v, s, np.nan)
    is_ex = (np.where(v, s, -999) == t[:, None]).any(axis=1)
    d = np.nanmin(np.abs(sv - t[:, None]), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    first = s[:, 0]
    last = s[np.arange(len(s)), np.maximum(n - 1, 0)]
    dl = np.abs(t - last); dfi = np.abs(t - first)
    base = (n >= 2) & ~is_ex
    return rate(base & (dl <= 5) & (dfi > 5)) - rate(base & (dfi <= 5) & (dl > 5))
