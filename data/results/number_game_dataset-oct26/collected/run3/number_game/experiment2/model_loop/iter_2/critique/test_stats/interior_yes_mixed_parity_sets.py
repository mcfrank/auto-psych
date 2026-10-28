# name: interior_yes_mixed_parity_sets
# description: For sets of 2+ examples with mixed parity (no parity rule) and range >= 15, yes-rate on non-example targets strictly inside the examples' min-max range; observed < model means people fill in the range between scattered examples less than the model's interval concepts predict, observed > model means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.mod(ex, 2)
    mixed = (np.nanmax(par, axis=1) != np.nanmin(par, axis=1))
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & mixed & (hi - lo >= 15) & (t > lo) & (t < hi) & ~is_ex
    return float(y[m].mean()) if m.any() else 0.0
