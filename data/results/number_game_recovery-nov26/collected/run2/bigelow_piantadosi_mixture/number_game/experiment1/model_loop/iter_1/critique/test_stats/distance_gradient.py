# name: distance_gradient
# description: Yes rate for targets within 3 of the nearest example minus yes rate for targets >=15 from the nearest example (targets equal to an example excluded); observed below null means the model over-produces a numerical-similarity (interval) gradient that people don't show, above means people rely more on magnitude proximity.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[:, None]
    d = np.nanmin(np.abs(ex - t), axis=1)
    y = df["rating"].to_numpy().astype(float)
    near = (d >= 1) & (d <= 3)
    far = d >= 15
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(y[near].mean() - y[far].mean())
