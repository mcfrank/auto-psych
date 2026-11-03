# name: wide_set_interior_yes_rate
# description: For sets spanning at least 40, yes-rate for interior targets more than 8 from every example; observed below null means the model over-generalises to the whole interval between widely spread examples.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    m = ((hi - lo) >= 40) & inr & (dist > 8)
    return float(r[m].mean()) if m.any() else 0.0
