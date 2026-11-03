# name: single_example_near_large_minus_small
# description: For single-example sets, yes rate on non-example targets within 10 of the example when the example is above 50 minus when it is 50 or below; observed above null means people generalize more broadly around large numbers than the model's Weber width predicts, below means less.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    one = (n == 1) & (nd <= 10) & ~isin
    v = r[one & (mn > 50)].mean() - r[one & (mn <= 50)].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
