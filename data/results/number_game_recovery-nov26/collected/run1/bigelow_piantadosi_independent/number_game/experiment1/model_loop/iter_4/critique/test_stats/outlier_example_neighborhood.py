# name: outlier_example_neighborhood
# description: For sets of >=3 examples where exactly one example breaks the majority last digit, yes rate among non-example targets within 3 of that outlier that do not end in the majority digit; observed below null means humans discount the exceptional example (do not generalise around it) more than the model, which places it inside a covering interval, above null means they treat it as a genuine member more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    ld = np.where(valid, np.mod(ex, 10), np.nan)
    cnt = np.stack([(ld == d).sum(axis=1) for d in range(10)], axis=1)
    maj = cnt.argmax(axis=1); majc = cnt.max(axis=1)
    is_out = valid & (ld != maj[:, None])
    dout = np.where(is_out, np.abs(ex - t[:, None]), np.inf).min(axis=1)
    m = (n >= 3) & (majc == n - 1) & nonmember & (dout <= 3) & (np.mod(t, 10) != maj)
    return float(r[m].mean()) if m.any() else 0.0
