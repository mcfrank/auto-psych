# name: parity_outlier_neighbour_yes
# description: In sets of >= 3 examples where exactly one example differs in parity from all others, yes rate for non-example targets within 3 of that off-parity example; observed > null means people generalise around the exception (treat it as an interval seed) more than the rule-plus-exception model, observed < null means they reject its neighbours more firmly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    odd = np.where(np.isnan(ex), np.nan, ex % 2)
    n_odd = np.nansum(odd, axis=1)
    lone_odd = (n >= 3) & (n_odd == 1)
    lone_even = (n >= 3) & (n_odd == n - 1)
    want = np.where(lone_odd, 1.0, np.where(lone_even, 0.0, -1.0))
    is_out = (odd == want[:, None])
    d_out = np.nanmin(np.where(is_out, np.abs(ex - t[:, None]), np.nan), axis=1) if is_out.any() else np.full(len(t), np.nan)
    m = (lone_odd | lone_even) & nonex & (np.nan_to_num(d_out, nan=99.0) <= 3)
    return rate(m)
