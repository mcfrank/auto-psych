# name: proximity_gradient
# description: Yes rate among non-example targets within 3 of the nearest example minus among those 20+ away; observed above null means human generalisation falls off with numerical distance more steeply than the model's interval hypotheses predict, below null means less steeply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    a, b = r[nonmember & (nearest <= 3)], r[nonmember & (nearest >= 20)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
