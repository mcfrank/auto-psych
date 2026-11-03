# name: parity_consistency_gap
# description: Among sets with >=2 examples all of one parity, yes rate for non-example targets of the same parity minus targets of the opposite parity; observed above null_mean means people use parity more than the model (which under-weights odd/even rules), below means the model over-uses parity.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    par = np.mod(E, 2)
    p0 = par[:, 0]
    same = np.all((par == p0[:, None]) | np.isnan(par), axis=1) & (n_ex >= 2)
    notex = ~np.any(E == t[:, None], axis=1)
    tp = np.mod(t, 2) == p0
    a = same & notex & tp
    b = same & notex & ~tp
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
