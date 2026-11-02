# name: magnitude_scaled_neighbour_contrast
# description: Yes rate for non-example targets 2-6 from the nearest example when that example is >= 50, minus the same when it is <= 20; observed > null means generalisation widens with magnitude (log-like number line) more than the model's compression predicts, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    idx = np.nanargmin(np.abs(ex - t[:, None]), axis=1)
    near = ex[np.arange(len(t)), idx]
    band = (d >= 2) & (d <= 6)
    a = band & (near >= 50)
    b = band & (near <= 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
