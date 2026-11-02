# name: shared_last_digit_target_yes
# description: For 2+ example sets whose examples all end in the same digit, yes rate on non-example targets ending in that digit; observed > null means people extend the "ends in d" rule more than the model (model under-weights final-digit rules), observed < null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    ld = ex % 10
    same = (ld.eq(ld[0], axis=0) | ex.isna()).all(axis=1) & (n >= 2)
    inset = ex.eq(t, axis=0).any(axis=1)
    sel = same & ~inset & ((t % 10) == ld[0])
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
