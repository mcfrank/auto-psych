# name: parity_congruence_within_span
# description: For sets of >=2 examples all of one parity, among targets inside [min,max] of the examples and not in the set, yes rate for same-parity targets minus opposite-parity targets; observed > null means people use parity within the magnitude range more strongly than the model (which only gets parity through whole-range odd/even rules) predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    par = np.where(np.isnan(E), np.nan, E % 2)
    pmin = np.nanmin(par, axis=1); pmax = np.nanmax(par, axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    ok = (n >= 2) & (pmin == pmax) & (t > lo) & (t < hi) & ~in_set
    same = (t % 2) == pmin
    r = df["rating"].to_numpy().astype(float)
    a = ok & same; b = ok & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
