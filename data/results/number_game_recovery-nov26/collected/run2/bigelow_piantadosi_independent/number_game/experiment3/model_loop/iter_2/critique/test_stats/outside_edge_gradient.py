# name: outside_edge_gradient
# description: For 2+-example sets, among out-of-span targets, yes-rate when the target is 1-4 beyond the nearest span edge minus when it is 15+ beyond; observed above null_mean means generalization falls off just outside the examples' span more steeply than the model's interval posterior produces (below = shallower, i.e. people extend further).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    y = df["rating"].astype(float).to_numpy()
    near = (n >= 2) & (dist >= 1) & (dist <= 4)
    far = (n >= 2) & (dist >= 15)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(y[near].mean() - y[far].mean())
