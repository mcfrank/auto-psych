# name: shared_digit_far_targets
# description: Among targets more than 15 from every example, yes rate when the target shares any digit with any example minus when it shares none; observed above null means people use loose digit-feature similarity beyond the model's explicit digit rules.
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
    ex = df["number_set"].astype(str).str.replace(",", "", regex=False)
    ts = t.astype(str)
    sh = pd.Series(False, index=df.index)
    for d in "0123456789":
        sh = sh | (ex.str.contains(d, regex=False) & ts.str.contains(d, regex=False))
    far = nd > 15
    v = r[far & sh].mean() - r[far & ~sh].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
