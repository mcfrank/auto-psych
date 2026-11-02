# name: same_last_digit_set_offrule_inspan_rate
# description: For sets of >=3 examples that all end in the same digit, yes rate on targets inside the example span that do NOT end in that digit; observed > null means people fall back on the range more than the model (model over-commits to the ends-in-d rule), observed < null means they follow the rule more strictly.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    k = ex.notna().sum(axis=1)
    ld = ex % 10
    same = (ld.max(axis=1) == ld.min(axis=1)) & (k >= 3)
    t = df["target"].astype(int)
    inspan = (t > ex.min(axis=1)) & (t < ex.max(axis=1))
    m = same & inspan & ((t % 10) != ld.min(axis=1))
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
