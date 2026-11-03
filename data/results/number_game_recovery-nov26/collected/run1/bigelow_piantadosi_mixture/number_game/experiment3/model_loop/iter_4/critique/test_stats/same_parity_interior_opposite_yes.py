# name: same_parity_interior_opposite_yes
# description: For sets of >=3 examples all of one parity, yes rate on opposite-parity targets lying strictly inside the examples' span; observed > null means people include parity-violating numbers inside the range more than the model (which lets the parity rule dominate), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1); span = hi - lo
    dnear = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    is_ex = dnear == 0
    beyond = np.maximum(0, np.maximum(lo - t, t - hi))
    interior = (t > lo) & (t < hi) & ~is_ex
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    e = np.nan_to_num(ex, nan=0).astype(int); valid = ~np.isnan(ex)
    n_odd = np.sum(valid & (e % 2 == 1), axis=1)
    all_odd = n_odd == k; all_even = n_odd == 0
    tt = t.astype(int)
    opp = (all_odd & (tt % 2 == 0)) | (all_even & (tt % 2 == 1))
    return rate(interior & (k >= 3) & opp)
