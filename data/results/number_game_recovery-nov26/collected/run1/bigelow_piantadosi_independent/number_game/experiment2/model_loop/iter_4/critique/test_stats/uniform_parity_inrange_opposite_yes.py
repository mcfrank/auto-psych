# name: uniform_parity_inrange_opposite_yes
# description: In sets of >= 3 examples that all share one parity, yes rate for opposite-parity targets lying strictly between the smallest and largest example; observed > null means people let the examples' range (similarity) override the parity rule more than modelled, observed < null means they apply the parity rule more strictly than modelled.
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
    uni = (n >= 3) & ((odd == n) | (even == n))
    setp = np.where(odd == n, 1, 0)
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    m = uni & (np.mod(t, 2) != setp) & (t > lo) & (t < hi)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
