# name: two_example_far_extrapolation
# description: For two-example sets, yes rate for targets more than 15 beyond the nearer end of the span (above max+15 or below min-15); observed above null means people extrapolate far beyond a pair of examples more than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    m = (n == 2) & ((t > mx + 15) | (t < mn - 15))
    v = r[m].mean()
    return float(0.0 if np.isnan(v) else v)
