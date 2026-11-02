# name: single_example_generalisation
# description: Yes-rate among single-example sets for targets other than the example; observed above null means humans generalise more broadly from one example than the model, below null means more narrowly.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    m = (n == 1) & (dist > 0)
    return float(r[m].mean()) if m.any() else 0.0
