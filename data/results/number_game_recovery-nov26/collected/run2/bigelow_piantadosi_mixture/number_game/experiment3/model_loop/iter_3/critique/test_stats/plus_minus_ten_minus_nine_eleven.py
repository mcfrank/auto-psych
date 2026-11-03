# name: plus_minus_ten_minus_nine_eleven
# description: Yes rate on non-example targets exactly 10 from some example minus those exactly 9 or 11 from some example (and not 10 from any); observed above null means people generalize along same-final-digit neighbours in the next decade more than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    dd = x.sub(t, axis=0).abs()
    ten = dd.eq(10).any(axis=1)
    ne = (dd.eq(9) | dd.eq(11)).any(axis=1) & ~ten
    v = r[ten & ~isin].mean() - r[ne & ~isin].mean()
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
