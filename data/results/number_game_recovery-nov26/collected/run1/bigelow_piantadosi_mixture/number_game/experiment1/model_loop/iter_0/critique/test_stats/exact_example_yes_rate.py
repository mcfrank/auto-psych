# name: exact_example_yes_rate
# description: Yes-rate when the target equals one of the shown examples; observed below null means the model over-predicts certain 'yes' for repeated examples (humans are less than deterministic).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    m = dist == 0
    return float(r[m].mean()) if m.any() else 0.0
