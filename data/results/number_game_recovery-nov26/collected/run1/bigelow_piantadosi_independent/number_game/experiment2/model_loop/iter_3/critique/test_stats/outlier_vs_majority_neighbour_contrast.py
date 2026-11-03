# name: outlier_vs_majority_neighbour_contrast
# description: In sets of >= 3 examples with exactly one minority-parity example, yes rate for non-example targets within 3 of the outlier minus yes rate for minority-parity (rule-violating) non-example targets within 3 of a majority example; observed > null means the outlier's neighbourhood is privileged over mere proximity to other examples more than modelled, observed < null means people treat proximity to any example alike (interval-like) more than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    par = np.mod(ex, 2)
    odd = np.nansum(par == 1, axis=1)
    even = np.nansum(par == 0, axis=1)
    one_out = (n >= 3) & ((odd == 1) | (even == 1))
    minp = np.where(odd == 1, 1, 0)
    is_out = (par == minp[:, None])
    dist = np.abs(ex - t[:, None])
    d_out = np.nanmin(np.where(is_out, dist, np.inf), axis=1)
    d_maj = np.nanmin(np.where(~is_out & ~np.isnan(ex), dist, np.inf), axis=1)
    a = one_out & (d > 0) & (d_out <= 3)
    b = one_out & (d > 0) & (d_out > 3) & (d_maj <= 3) & (np.mod(t, 2) == minp)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
