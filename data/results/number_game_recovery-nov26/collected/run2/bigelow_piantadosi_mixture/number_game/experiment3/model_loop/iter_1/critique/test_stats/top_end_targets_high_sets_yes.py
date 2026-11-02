# name: top_end_targets_high_sets_yes
# description: Yes rate for non-example targets 91-100 on sets whose largest example is >= 80; observed below null means people extend high clusters toward the top of the scale less than the Weber-width/heading model predicts, above means more.
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
    m = ~isin & (t > 90) & (mx >= 80)
    v = r[m].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
