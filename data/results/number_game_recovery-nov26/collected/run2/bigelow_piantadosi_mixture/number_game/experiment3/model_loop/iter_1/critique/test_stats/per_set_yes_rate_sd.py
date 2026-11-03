# name: per_set_yes_rate_sd
# description: Standard deviation across number_sets of the per-set yes rate; observed above null means sets differ in generalization breadth more (model too uniform across sets), below means the model exaggerates set-to-set differences.
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
    v = r.groupby(df["number_set"].astype(str)).mean().std()
    return float(0.0 if (v is None or np.isnan(v)) else v)
