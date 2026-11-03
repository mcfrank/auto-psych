# name: inrange_four_minus_two
# description: Yes-rate for non-example targets strictly inside the examples' range in 4-example sets minus in 2-example sets; observed > null means more examples raise in-range filling more for people than in the model (stronger evidence for a range), observed < null means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    P = parts.values
    n = parts.notna().sum(axis=1).values
    lo = np.nanmin(P, axis=1)
    hi = np.nanmax(P, axis=1)
    t = df["target"].astype(float).values
    isex = np.nansum(P == t[:, None], axis=1) > 0
    inr = (t > lo) & (t < hi) & ~isex
    y = df["rating"].values.astype(float)
    a, b = inr & (n == 4), inr & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
