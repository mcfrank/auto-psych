# name: shared_divisor_rule_yes_rate
# description: For 2+ example sets whose examples are all multiples of some m in 3..10, yes rate for non-example targets that are multiples of such a shared m; observed < null means people extend arithmetic (multiple-of-m) rules less than the model, observed > null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    hit = pd.Series(False, index=df.index)
    for mm in range(3, 11):
        allm = ((ex % mm == 0) | ex.isna()).all(axis=1) & (n >= 2)
        hit = hit | (allm & (t % mm == 0))
    sel = hit & ~inset
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
