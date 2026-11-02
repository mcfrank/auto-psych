# name: interior_yes_setsize_contrast
# description: For sets spanning >=20, yes rate for unshown targets strictly inside the span with 4 examples minus with 2 examples; observed above null means extra examples fill in the range more than the model's size principle predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    shown = (valid & (ex == t[:, None])).any(axis=1)
    inside = (t > lo) & (t < hi) & ~shown & (hi - lo >= 20)
    a, b = inside & (n == 4), inside & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
