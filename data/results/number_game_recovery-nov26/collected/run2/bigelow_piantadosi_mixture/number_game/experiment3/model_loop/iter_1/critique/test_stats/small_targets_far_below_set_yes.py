# name: small_targets_far_below_set_yes
# description: Yes rate for non-example targets 1-10 on sets whose smallest example is >= 30; observed above null means people endorse small numbers far below the examples more than the model predicts (a small-number bias), below means less.
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
    m = ~isin & (t <= 10) & (mn >= 30)
    v = r[m].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
