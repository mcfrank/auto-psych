# name: outlier_beyond_radius_yes
# description: In sets of >= 3 examples with exactly one minority-parity example, yes rate for minority-parity non-example targets 4-9 away from that outlier and more than 3 from every majority example; observed > null means the outlier's exception region extends farther than the model's (radius <= 3) neighbourhood, observed < null means it is narrower.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    d = dist.min(axis=1)
    par = np.where(valid, np.mod(ex, 2), -1)
    odd = (par == 1).sum(axis=1)
    even = (par == 0).sum(axis=1)
    one_out = (n >= 3) & ((odd == 1) | (even == 1)) & (odd != even)
    minp = np.where(odd == 1, 1, 0)
    is_out = par == minp[:, None]
    d_out = np.where(is_out, dist, np.inf).min(axis=1)
    d_maj = np.where(~is_out & valid, dist, np.inf).min(axis=1)
    m = one_out & (d > 0) & (d_out >= 4) & (d_out <= 9) & (d_maj > 3) & (np.mod(t, 2) == minp)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
