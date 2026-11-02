# name: double_or_half_of_example_yes
# description: Yes rate for non-example targets equal to twice or half of a shown example; observed above null means people use multiplicative (doubling/halving) relations more than the model predicts.
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
    m = (x.mul(2).eq(t, axis=0) | x.div(2).eq(t, axis=0)).any(axis=1) & ~isin
    v = r[m].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
