# name: unstructured_wide_set_nonmember_rate
# description: Yes rate on non-example targets for sets of >=3 examples spanning >=60 that share no parity, no last digit and no divisor 3..12; observed > null means people accept more numbers than the model when the examples show no rule, observed < null means they reject more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    k = ex.notna().sum(axis=1)
    span = ex.max(axis=1) - ex.min(axis=1)
    shared = pd.Series(False, index=df.index)
    for m in range(2, 13):
        shared |= ((ex % m == 0) | ex.isna()).all(axis=1)
    shared |= ((ex % 2 == 1) | ex.isna()).all(axis=1)
    ld = ex % 10
    shared |= (ld.max(axis=1) == ld.min(axis=1))
    t = df["target"].astype(int)
    inset = ex.eq(t, axis=0).any(axis=1)
    m = (k >= 3) & (span >= 60) & ~shared & ~inset
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
