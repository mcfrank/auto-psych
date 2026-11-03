# name: square_set_square_target_lift
# description: For sets of two or more examples that are all perfect squares, yes rate on non-example square targets minus yes rate on non-square targets; observed above null means the model's extra squares cost suppresses the 'perfect squares' reading more than people do, and below null means squares are even less available than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    n = ex.notna().sum(axis=1)
    S = [k * k for k in range(1, 11)]
    allsq = (ex.isin(S) | ex.isna()).all(axis=1) & (n >= 2)
    sq = t.isin(S)
    a = (allsq & sq & (mind > 0)).values
    b = (allsq & ~sq).values
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
