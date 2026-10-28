# name: parity_match_gap
# description: Among sets of >=2 examples that are all even or all odd but do NOT share a last digit, yes-rate for non-member targets of the same parity minus targets of opposite parity; observed above null means people use the parity (even/odd) rule more than the model predicts.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    t = df["target"].astype(int)
    par = s.mod(2)
    p0 = par[0]
    samepar = (par.eq(p0, axis=0) | par.isna()).all(axis=1)
    ld = s.mod(10)
    sameld = (ld.eq(ld[0], axis=0) | ld.isna()).all(axis=1)
    ok = samepar & ~sameld & (n >= 2) & ~s.eq(t, axis=0).any(axis=1)
    match = ok & (t % 2 == p0)
    miss = ok & (t % 2 != p0)
    r = df["rating"].astype(float)
    if not match.any() or not miss.any():
        return 0.0
    return float(r[match].mean() - r[miss].mean())
