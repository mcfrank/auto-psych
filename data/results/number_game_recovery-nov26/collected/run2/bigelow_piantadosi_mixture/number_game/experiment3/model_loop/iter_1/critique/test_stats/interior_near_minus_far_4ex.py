# name: interior_near_minus_far_4ex
# description: For 4-example sets, among non-example targets strictly inside the example span, yes rate for targets within 3 of the nearest example minus targets >= 10 from every example; observed above null means people show an exemplar-like similarity gradient inside the span that the range model flattens, below means interior gaps are filled more than the model predicts.
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
    m = (n == 4) & ~isin & (t > mn) & (t < mx)
    a = r[m & (nd <= 3)].mean()
    b = r[m & (nd >= 10)].mean()
    v = a - b
    return float(0.0 if (v is None or np.isnan(v)) else v)
