# name: near_outside_range_yes_rate
# description: Yes-rate for targets outside the example range but within 10 of the nearest example; observed above null means humans extend generalisation beyond the examples more than the model's interval/size-principle posterior allows (below null: less).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    m = (~inr) & (dist <= 10)
    return float(r[m].mean()) if m.any() else 0.0
