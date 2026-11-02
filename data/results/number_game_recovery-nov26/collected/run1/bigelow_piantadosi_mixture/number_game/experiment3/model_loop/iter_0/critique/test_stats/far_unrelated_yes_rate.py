# name: far_unrelated_yes_rate
# description: Yes rate for "unrelated" targets: >=25 away from every example, sharing no last digit with any example, and of opposite parity when all examples share a parity; observed > model means the model under-produces baseline/lapse yeses far from the examples.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tg = df["target"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    d = np.where(valid, np.abs(ex - tg[:, None]), np.inf).min(axis=1)
    same_last = (valid & (np.mod(ex, 10) == np.mod(tg, 10)[:, None])).any(axis=1)
    par = np.mod(ex, 2)
    n = valid.sum(axis=1)
    n_odd = np.where(valid, par, 0).sum(axis=1)
    all_even = n_odd == 0
    all_odd = n_odd == n
    tpar = np.mod(tg, 2)
    parity_ok = ~((all_even & (tpar == 0)) | (all_odd & (tpar == 1)))
    sel = (d >= 25) & (~same_last) & parity_ok
    r = df["rating"].astype(float).to_numpy()
    return float(r[sel].mean()) if sel.any() else 0.0
