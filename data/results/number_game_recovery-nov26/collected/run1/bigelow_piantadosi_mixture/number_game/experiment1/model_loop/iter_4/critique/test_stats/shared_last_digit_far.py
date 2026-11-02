# name: shared_last_digit_far
# description: For targets more than 15 outside the examples' range, yes-rate when the target shares its last digit with some example minus when it shares none; observed < null means the model over-generalises by "ends in d" rules relative to people, observed > null means people use last-digit similarity more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    P = parts.values
    lo = np.nanmin(P, axis=1)
    hi = np.nanmax(P, axis=1)
    t = df["target"].astype(float).values
    gap = np.maximum(lo - t, t - hi)
    far = gap > 15
    share = np.nansum((np.mod(P, 10) == np.mod(t, 10)[:, None]), axis=1) > 0
    y = df["rating"].values.astype(float)
    a, b = far & share, far & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
