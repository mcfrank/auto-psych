# name: gradient_near_minus_mid
# description: Yes-rate for targets 1-5 beyond the examples' range minus yes-rate for targets 11-25 beyond it; observed > null means human generalisation falls off more steeply just outside the range than the model's Erlang interval prior produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    d = np.maximum(np.nanmin(ex, axis=1) - t, t - np.nanmax(ex, axis=1))
    near = (d >= 1) & (d <= 5); mid = (d >= 11) & (d <= 25)
    a = r[near].mean() if near.any() else 0.0; b = r[mid].mean() if mid.any() else 0.0
    return float(a - b)
