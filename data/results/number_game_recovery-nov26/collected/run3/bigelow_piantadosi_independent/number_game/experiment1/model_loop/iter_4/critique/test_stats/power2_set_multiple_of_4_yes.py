# name: power2_set_multiple_of_4_yes
# description: For 2+ example sets made only of powers of 2, yes rate on non-example targets that are multiples of 4; observed > null means people generalise from power sets to multiples-of-4 even more than the model (exponent penalty not strong enough), observed < null means the model over-extends to multiples.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    allp = (ex.isin([1, 2, 4, 8, 16, 32, 64]) | ex.isna()).all(axis=1) & (n >= 2)
    inset = ex.eq(t, axis=0).any(axis=1)
    sel = allp & ~inset & (t % 4 == 0)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
