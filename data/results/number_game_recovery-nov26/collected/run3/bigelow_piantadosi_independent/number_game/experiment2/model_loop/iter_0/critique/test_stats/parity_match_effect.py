# name: parity_match_effect
# description: For sets whose examples are all even or all odd, yes rate on non-example targets of the same parity minus yes rate on targets of the opposite parity; observed below null means the model over-uses odd/even concepts (it predicts too much parity-driven generalization), and above null means it under-uses them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    par = ex % 2
    alleven = (par.fillna(0) == 0).all(axis=1)
    allodd = (par.fillna(1) == 1).all(axis=1)
    same = ((alleven & (t % 2 == 0)) | (allodd & (t % 2 == 1))) & (mind > 0)
    opp = (alleven & (t % 2 == 1)) | (allodd & (t % 2 == 0))
    same, opp = same.values, opp.values
    if same.sum() == 0 or opp.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[opp].mean())
