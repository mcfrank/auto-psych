# name: parity_only_far_target_yes
# description: For 2+ example sets of uniform parity, yes rate on non-example same-parity targets that are >10 from every example, share no digit with any example, and are not multiples of any m in 3..10 dividing all examples (only "even"/"odd" licenses them); observed < null means the model over-uses bare parity rules, observed > null means it under-uses them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    par = ex % 2
    unif = (par.eq(par[0], axis=0) | ex.isna()).all(axis=1) & (n >= 2)
    samepar = (t % 2) == par[0]
    dist = ex.sub(t, axis=0).abs().min(axis=1)
    def digs(x):
        tens = (x // 10) % 10
        return [x % 10, tens.where(x >= 10)]
    td = digs(t)
    sharedig = pd.Series(False, index=df.index)
    for col in ex.columns:
        for a in digs(ex[col]):
            for b in td:
                sharedig = sharedig | (a == b)
    mult = pd.Series(False, index=df.index)
    for mm in range(3, 11):
        allm = ((ex % mm == 0) | ex.isna()).all(axis=1)
        mult = mult | (allm & (t % mm == 0))
    sel = unif & samepar & ~inset & (dist > 10) & ~sharedig & ~mult
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
