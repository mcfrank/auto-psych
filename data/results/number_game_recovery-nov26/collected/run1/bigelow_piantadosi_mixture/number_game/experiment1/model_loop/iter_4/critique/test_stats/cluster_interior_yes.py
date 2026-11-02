# name: cluster_interior_yes
# description: For 3-4-example sets spanning <= 40, yes-rate for non-example targets strictly inside the examples' range; observed < null means people fill in a compact cluster's interior less than the model's range guesses predict, observed > null means more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    P = parts.values
    n = parts.notna().sum(axis=1).values
    lo = np.nanmin(P, axis=1)
    hi = np.nanmax(P, axis=1)
    t = df["target"].astype(float).values
    isex = np.nansum(P == t[:, None], axis=1) > 0
    m = (n >= 3) & (hi - lo <= 40) & (t > lo) & (t < hi) & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
