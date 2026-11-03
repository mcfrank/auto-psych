# name: exception_set_majority_digit_far
# description: Sets of 3-4 examples in which all but exactly one example share a last digit d: yes-rate for non-example targets ending in d that are >=10 from every example; observed above null_mean means people keep the digit rule despite one exception more than the model (trust/stickiness too low), below means the model over-tolerates the exception.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    best = np.zeros(len(t)); dig = np.full(len(t), -1.0)
    for k in range(10):
        c = np.sum(ld == k, axis=1)
        upd = c > best
        dig = np.where(upd, k, dig); best = np.where(upd, c, best)
    m = (n >= 3) & (best == n - 1) & ~isex & (np.mod(t, 10) == dig) & (dmin >= 10)
    return float(r[m].mean()) if m.sum() > 0 else 0.0
