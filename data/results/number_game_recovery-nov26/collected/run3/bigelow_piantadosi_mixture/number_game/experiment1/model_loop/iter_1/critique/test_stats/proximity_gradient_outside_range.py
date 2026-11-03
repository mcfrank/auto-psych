# name: proximity_gradient_outside_range
# description: For targets outside the examples' range, yes rate at distance 1-5 from the nearest example minus yes rate at distance 11-30; observed > null means people show a steeper similarity/proximity gradient when extrapolating than the model's interval hypotheses produce.
def test_statistic(df):
    a = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    mind = np.nanmin(np.abs(a - t[:, None]), axis=1)
    out = (t < np.nanmin(a, axis=1)) | (t > np.nanmax(a, axis=1))
    near = out & (mind >= 1) & (mind <= 5)
    far = out & (mind >= 11) & (mind <= 30)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
