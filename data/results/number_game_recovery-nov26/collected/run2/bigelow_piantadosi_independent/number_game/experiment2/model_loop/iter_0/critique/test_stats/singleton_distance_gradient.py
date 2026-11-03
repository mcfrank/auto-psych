# name: singleton_distance_gradient
# description: For single-example sets, yes rate for targets within 10 of the example (excluding it) minus targets more than 30 away; observed above null means a steeper similarity gradient for one example than the model predicts, below means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    s = (n == 1)
    d = np.abs(t - ex[:, 0])
    return rate(s & (d >= 1) & (d <= 10)) - rate(s & (d > 30))
