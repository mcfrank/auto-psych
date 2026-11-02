# name: single_example_distance_gradient
# description: For one-example sets, yes rate for targets within 10 of the example (excluding the example) minus yes rate for targets more than 30 away; observed above null means a steeper similarity falloff from a lone example than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    d = (t - x[0]).abs()
    one = (n == 1) & ~isin
    a = r[one & (d <= 10)].mean()
    b = r[one & (d > 30)].mean()
    return float(0.0 if (np.isnan(a) or np.isnan(b)) else a - b)
