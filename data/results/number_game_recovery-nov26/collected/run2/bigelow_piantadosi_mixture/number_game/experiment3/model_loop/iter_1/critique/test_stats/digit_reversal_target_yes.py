# name: digit_reversal_target_yes
# description: Yes rate for non-example two-digit targets that are the digit reversal of a shown example (e.g. 38 for 83); observed above null means people generalize by digit reversal more than the model (which has no reversal concept) predicts.
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
    rev = (t % 10) * 10 + t // 10
    m = x.eq(rev, axis=0).any(axis=1) & ~isin & (t >= 10) & (t % 10 > 0)
    v = r[m].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
