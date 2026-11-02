# name: partial_final_digit_far_same_parity
# description: In sets of >= 2 examples sharing one parity but not all ending in the same digit, among same-parity non-example targets >= 5 from every example, yes rate when the target's final digit matches some example's final digit minus when it matches none; observed > null means people generalize by partial final-digit resemblance that no single rule in the model captures, observed < null means less than modelled.
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
    uni = (n >= 2) & ((odd == n) | (even == n))
    setp = np.where(odd == n, 1, 0)
    fd = np.where(valid, np.mod(ex, 10), -1)
    fdmax = fd.max(axis=1)
    fdmin = np.where(valid, fd, 99).min(axis=1)
    mixed = fdmax != fdmin
    share = (fd == np.mod(t, 10)[:, None]).any(axis=1)
    base = uni & mixed & (np.mod(t, 2) == setp) & (d >= 5)
    a, b = base & share, base & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
