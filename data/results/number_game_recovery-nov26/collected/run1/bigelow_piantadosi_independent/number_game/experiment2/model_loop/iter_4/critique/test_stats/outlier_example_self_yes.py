# name: outlier_example_self_yes
# description: In sets of >= 3 examples with exactly one minority-parity example, yes rate when the target is that outlier example itself; observed < null means people reject the anomalous example (treat it as noise) more than the model's exception concepts predict, observed > null means they accept it more.
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
    hit = (is_out & (ex == t[:, None])).any(axis=1)
    m = one_out & hit
    return float(y[m].mean()) if m.sum() > 0 else 0.0
