# name: adjacent_minus_nearby_targets
# description: Yes rate on non-example targets exactly 1 from the nearest example minus yes rate on targets 3-5 from the nearest example; observed above null means generalization falls off more steeply right next to the examples than the model predicts, below means flatter.
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
    v = r[(nd == 1) & ~isin].mean() - r[(nd >= 3) & (nd <= 5)].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
