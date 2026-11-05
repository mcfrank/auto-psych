# name: interval_inside_vs_outside_near
# description: In size>=2 sets, yes rate for unshown targets strictly inside [min,max] minus targets outside the range within 1-10 of nearest example; observed above null means the model under-weights interval/range-based generalisation.
def test_statistic(df):
    S = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(S), axis=1)
    Si = np.nan_to_num(S, nan=0.0).astype(int)
    dist = np.nanmin(np.abs(S - t[:, None]), axis=1)
    smin = np.nanmin(S, axis=1); smax = np.nanmax(S, axis=1)
    def m(mask):
        return float(np.mean(r[mask])) if np.sum(mask) > 0 else 0.0
    sel = n >= 2
    inside = (t > smin) & (t < smax) & (dist > 0)
    outside = ((t < smin) | (t > smax)) & (dist >= 1) & (dist <= 10)
    return m(sel & inside) - m(sel & outside)
