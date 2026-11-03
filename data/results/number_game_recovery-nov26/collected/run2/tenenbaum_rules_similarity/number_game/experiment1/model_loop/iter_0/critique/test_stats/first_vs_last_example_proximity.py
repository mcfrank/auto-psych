# name: first_vs_last_example_proximity
# description: Among non-member targets of multi-example sets, corr(rating, -|target - first shown example|) minus corr(rating, -|target - last shown example|); nonzero observed vs a near-zero null means order of presentation matters (primacy if positive, recency if negative), which the order-invariant model cannot produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    n = valid.sum(axis=1)
    m = (~member) & (n >= 2)
    if m.sum() < 3: return 0.0
    first = ex[m, 0]
    last = ex[m][np.arange(m.sum()), n[m] - 1]
    tt = t[m, 0]; y = r[m]
    d1 = -np.abs(tt - first); d2 = -np.abs(tt - last)
    if y.std() == 0 or d1.std() == 0 or d2.std() == 0: return 0.0
    return float(np.corrcoef(d1, y)[0, 1] - np.corrcoef(d2, y)[0, 1])
