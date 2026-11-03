# name: primacy_near_first_minus_last
# description: In multi-example sets, yes-rate for targets within 5 of the first-shown example (and not within 5 of any other) minus that for targets within 5 of the last-shown example only; nonzero observed vs a ~0 null means example order matters, which the order-blind model cannot produce.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = s.str.count(",").to_numpy() + 1
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None]); dist[np.isnan(dist)] = 1e9
    idx = np.arange(len(t)); last = dist[idx, np.clip(n - 1, 0, ex.shape[1] - 1)]
    first = dist[:, 0]; close = (dist <= 5).sum(axis=1)
    mf = (n >= 2) & (first <= 5) & (first > 0) & (close == 1)
    ml = (n >= 2) & (last <= 5) & (last > 0) & (close == 1)
    a = r[mf].mean() if mf.any() else 0.0; b = r[ml].mean() if ml.any() else 0.0
    return float(a - b)
