# name: powers_of_two_extension_contrast
# description: In sets whose examples are all powers of 2, yes rate on non-example powers of 2 minus yes rate on other even targets; observed > null means people extend the powers-of-2 rule more selectively than the model, observed < null means the model over-commits to the rule.
def test_statistic(df):
    p2 = {2, 4, 8, 16, 32, 64}
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    allp2 = ex.isin(p2) | ex.isna()
    sel = allp2.all(axis=1) & (ex.notna().sum(axis=1) >= 2)
    t = df["target"].astype(int)
    inset = ex.eq(t, axis=0).any(axis=1)
    rule = sel & t.isin(p2) & ~inset
    other = sel & ~t.isin(p2) & (t % 2 == 0)
    if rule.sum() == 0 or other.sum() == 0:
        return 0.0
    return float(df.loc[rule, "rating"].mean() - df.loc[other, "rating"].mean())
