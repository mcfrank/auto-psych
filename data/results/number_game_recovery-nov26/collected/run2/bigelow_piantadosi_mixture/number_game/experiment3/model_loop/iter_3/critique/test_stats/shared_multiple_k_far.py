# name: shared_multiple_k_far
# description: For sets of 2+ examples that are all multiples of some k in 3..12, yes rate on non-example targets more than 10 from every example that are multiples of k minus those that are not; observed above null means people extend shared-divisor rules to distant numbers more than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    hit = pd.Series(False, index=df.index)
    use = pd.Series(False, index=df.index)
    for k in range(3, 13):
        allk = ((x % k == 0) | x.isna()).all(axis=1) & (n >= 2)
        use = use | allk
        hit = hit | (allk & (t % k == 0))
    far = (nd > 10) & ~isin & use
    v = r[far & hit].mean() - r[far & ~hit].mean()
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
