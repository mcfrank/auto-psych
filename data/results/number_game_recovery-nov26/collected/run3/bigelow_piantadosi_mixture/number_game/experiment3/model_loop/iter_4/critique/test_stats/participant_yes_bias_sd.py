# name: participant_yes_bias_sd
# description: SD across participants (trial_index resets) of the mean item-residual yes over all their trials; observed > null means people differ in an overall yes/no response bias that the model, which shares one parameter set across participants, does not produce.
def test_statistic(df):
    N = np.arange(1, 101)
    rl = [N % 2 == 1, N % 2 == 0,
          np.array([v > 1 and all(v % a for a in range(2, int(v ** 0.5) + 1)) for v in N]),
          np.isin(N, [k * k for k in range(1, 11)]), np.isin(N, [k ** 3 for k in range(1, 5)])]
    rl += [N % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rl += [np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]) for m in range(2, 11)]
    rl += [N % 10 == dd for dd in range(10)]
    rl += [np.array([str(dd) in str(v) for v in N]) for dd in range(10)]
    R = np.array(rl)
    rsize = R.sum(1)
    s = df["number_set"].astype(str).to_numpy()
    t = np.clip(df["target"].astype(int).to_numpy(), 1, 100)
    y = (df["rating"].astype(float).to_numpy() > 0.5).astype(float)
    ti = df["trial_index"].astype(int).to_numpy()
    uniq, inv = np.unique(s, return_inverse=True)
    S = len(uniq)
    n_ = np.zeros(S, int); lo_ = np.zeros(S, int); hi_ = np.zeros(S, int)
    small_ = np.full(S, 999); smallmask = np.zeros((S, 100), bool); exmask = np.zeros((S, 100), bool)
    for i, u in enumerate(uniq):
        a = np.array([int(v) for v in u.split(",") if v.strip() != ""])
        n_[i] = len(a); lo_[i] = a.min(); hi_[i] = a.max(); exmask[i, a - 1] = True
        cov = R[:, a - 1].all(1)
        if cov.any():
            j = np.argmin(np.where(cov, rsize, 999))
            small_[i] = rsize[j]; smallmask[i] = R[j]
    n = n_[inv]; lo = lo_[inv]; hi = hi_[inv]; small = small_[inv]
    in_small = smallmask[inv, t - 1]
    is_ex = exmask[inv, t - 1]
    inhull = (t >= lo) & (t <= hi)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    pid = np.cumsum(np.r_[1, (np.diff(ti) < 0).astype(int)]) - 1
    ikey = inv * 101 + t
    _, iinv = np.unique(ikey, return_inverse=True)
    imean = np.bincount(iinv, weights=y) / np.bincount(iinv)
    res = y - imean[iinv]
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    P = pid.max() + 1
    m = np.bincount(pid, weights=res, minlength=P) / np.maximum(np.bincount(pid, minlength=P), 1)
    return float(np.std(m))
