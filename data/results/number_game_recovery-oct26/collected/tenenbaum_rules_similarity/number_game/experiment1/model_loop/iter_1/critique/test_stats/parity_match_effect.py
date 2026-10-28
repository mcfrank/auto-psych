# name: parity_match_effect
# description: For sets whose examples all share one parity, among non-example targets within the examples range +/-10: yes rate for same-parity targets minus opposite-parity targets; observed > null_mean means people use parity more than the model, < means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    par = np.mod(s, 2)
    allsame = (np.nanmax(par, axis=1) == np.nanmin(par, axis=1)) & (n >= 2)
    notin = ~np.any(s == t[:, None], axis=1)
    near = (t >= mn - 10) & (t <= mx + 10)
    same = np.mod(t, 2) == par[:, 0]
    base = allsame & notin & near
    a = base & same; b = base & ~same
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
