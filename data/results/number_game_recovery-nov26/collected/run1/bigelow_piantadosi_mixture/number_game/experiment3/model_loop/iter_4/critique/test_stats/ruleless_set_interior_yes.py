# name: ruleless_set_interior_yes
# description: For sets of >=3 examples with no simple shared rule (mixed parity, no common last digit, no common multiple 3-12), yes rate on non-example targets inside the examples' span; observed < null means the model over-fills the interior of rule-less spread sets (people endorse an interval less fully than membership implies), observed > null the reverse.
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
    e = np.nan_to_num(ex, nan=-1).astype(int); valid = ~np.isnan(ex)
    par = np.where(valid, e % 2, -1)
    mixed = (np.nanmax(np.where(valid, par, np.nan), axis=1) != np.nanmin(np.where(valid, par, np.nan), axis=1))
    ld = np.where(valid, e % 10, np.nan)
    same_ld = np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)
    common = np.zeros(len(df), dtype=bool)
    for mlt in range(3, 13):
        common |= np.all(np.where(valid, e % mlt == 0, True), axis=1)
    m = interior & (k >= 3) & mixed & ~same_ld & ~common
    return rate(m)
