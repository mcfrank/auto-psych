# name: square_targets_given_square_sets
# description: Yes rate for perfect-square non-example targets when every example is a perfect square; observed above null means people generalize the squares concept more than the model predicts, below means less.
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
    SQ = [k * k for k in range(1, 11)]
    allsq = x.isin(SQ).sum(axis=1).eq(n)
    m = allsq & t.isin(SQ) & ~isin
    v = r[m].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
