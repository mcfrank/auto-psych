# name: same_parity_set_parity_match_yes
# description: For sets of 3+ examples that are all even or all odd, yes rate for non-example targets with that same parity; observed below null means the model over-applies the odd/even rule (e.g. 70,10,88,40 -> 48) relative to people.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    par = x.mod(2)
    allsame = (par.max(axis=1) == par.min(axis=1)) & (n >= 3)
    m = allsame & ((t % 2) == par[0]) & ~isin
    v = r[m].mean()
    return float(0.0 if np.isnan(v) else v)
