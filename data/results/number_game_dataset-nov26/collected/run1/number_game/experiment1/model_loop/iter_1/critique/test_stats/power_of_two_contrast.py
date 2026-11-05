# name: power_of_two_contrast
# description: In sets of size>=2 consisting only of powers of two, yes rate for unshown power-of-two targets minus other even targets (both not in set); observed above null means the model misses the powers-of-two rule.
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
    p2 = np.isin(Si, [1,2,4,8,16,32,64]) | np.isnan(S)
    sp = (n >= 2) & np.all(p2, axis=1)
    tp = np.isin(t, [1,2,4,8,16,32,64])
    return m(sp & tp & (dist > 0)) - m(sp & ~tp & (t % 2 == 0) & (dist > 0))
