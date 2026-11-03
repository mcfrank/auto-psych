# name: square_set_nonexample_yes
# description: For 2+ example sets made only of perfect squares (1..100), yes rate on non-example targets; observed > null means people generalise more broadly after square sets than the model (whose penalised square concept pushes it to other hypotheses), observed < null means less broadly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    allsq = (ex.isin([k * k for k in range(1, 11)]) | ex.isna()).all(axis=1) & (n >= 2)
    inset = ex.eq(t, axis=0).any(axis=1)
    sel = allsq & ~inset
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
