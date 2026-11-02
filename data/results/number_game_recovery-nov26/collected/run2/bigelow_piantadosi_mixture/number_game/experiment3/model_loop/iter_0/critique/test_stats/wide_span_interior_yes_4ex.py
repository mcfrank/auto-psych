# name: wide_span_interior_yes_4ex
# description: Yes rate for non-example targets strictly inside the span of 4-example sets whose span (max-min) is at least 50; observed above null means people fill in wide scattered sets as a broad range more than the model predicts, below means less.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    m = (n == 4) & ((mx - mn) >= 50) & (t > mn) & (t < mx) & ~isin
    v = r[m].mean()
    return float(0.0 if np.isnan(v) else v)
