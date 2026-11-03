# name: multiple_rule_beyond_span_contrast
# description: In sets of >=3 examples all divisible by some m in 3..12 (not all powers of 2), among targets outside the example span, yes rate on multiples of m minus non-multiples; observed > null means people extend arithmetic rules beyond the examples' range more than the model, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    k = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    outside = (t < lo) | (t > hi)
    p2 = (ex.isin({2, 4, 8, 16, 32, 64}) | ex.isna()).all(axis=1)
    base = (k >= 3) & outside & ~p2
    yes_r, yes_o = [], []
    used = pd.Series(False, index=df.index)
    for m in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]:
        allm = ((ex % m == 0) | ex.isna()).all(axis=1) & base & ~used
        used |= allm
        yes_r.append(df.loc[allm & (t % m == 0), "rating"])
        yes_o.append(df.loc[allm & (t % m != 0), "rating"])
    r, o = pd.concat(yes_r), pd.concat(yes_o)
    if len(r) == 0 or len(o) == 0:
        return 0.0
    return float(r.mean() - o.mean())
