# name: recency_nearest_last_minus_first
# description: Among non-example targets of 2+-example sets, yes-rate when the target's nearest example is the last-shown one minus yes-rate when it is the first-shown one; observed > model means a recency effect (generalization anchored on the most recent example) that the order-invariant model cannot produce, observed < model means a primacy effect.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(axis=1)
    tgt = df["target"].astype(float).values
    d = np.abs(ex - tgt[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    is_ex = (d == 0).any(axis=1)
    nearest = np.argmin(d, axis=1)
    dmin = d.min(axis=1)
    last_idx = n - 1
    d_last = d[np.arange(len(d)), last_idx]
    d_first = d[:, 0]
    valid = (n >= 2) & (~is_ex)
    near_last = valid & (d_last == dmin) & (d_first > dmin)
    near_first = valid & (d_first == dmin) & (d_last > dmin)
    r = df["rating"].astype(float).values
    if near_last.sum() == 0 or near_first.sum() == 0:
        return 0.0
    return float(r[near_last].mean() - r[near_first].mean())
