# name: order_direction_extension
# description: For monotonically ordered sets of >=3 examples, yes-rate for targets 1-20 beyond the LAST-shown example (continuing the order) minus targets 1-20 beyond the FIRST-shown example; observed > null means people extrapolate in the direction the sequence was presented, which the order-blind model cannot do.
def test_statistic(df):
    N = np.arange(1, 101)
    rl = [N % 2 == 1, N % 2 == 0,
          np.array([v > 1 and all(v % a for a in range(2, int(v ** 0.5) + 1)) for v in N]),
          np.isin(N, [k * k for k in range(1, 11)]), np.isin(N, [k ** 3 for k in range(1, 5)])]
    rl += [N % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rl += [np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]) for m in range(2, 11)]
    rl += [N % 10 == dd for dd in range(10)]
    rl += [np.array([str(dd) in str(v) for v in N]) for dd in range(10)]
    R = np.array(rl)  # (45 rules, 100) membership of the incumbent's rules
    rsize = R.sum(1)
    s = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    y = (df["rating"].astype(float).to_numpy() > 0.5).astype(float)
    uniq, inv = np.unique(s, return_inverse=True)
    S = len(uniq)
    n_ = np.zeros(S, int); lo_ = np.zeros(S, int); hi_ = np.zeros(S, int)
    first_ = np.zeros(S, int); lastx_ = np.zeros(S, int); mono_ = np.zeros(S, int)
    kmax_ = np.zeros(S, int); odd_ = np.zeros(S, bool)
    small2_ = np.full(S, 999); small_ = np.full(S, 999); smallmask = np.zeros((S, 100), bool); cover = np.zeros((S, 100), bool)
    digs = np.zeros((S, 10), bool)
    for i, u in enumerate(uniq):
        xs = [int(v) for v in u.split(",") if v.strip() != ""]
        a = np.array(xs)
        n_[i] = len(xs); lo_[i] = a.min(); hi_[i] = a.max(); first_[i] = xs[0]; lastx_[i] = xs[-1]
        if len(xs) >= 3 and np.all(np.diff(a) > 0): mono_[i] = 1
        if len(xs) >= 3 and np.all(np.diff(a) < 0): mono_[i] = -1
        ks = [k for k in range(2, 11) if np.all(a % k == 0)]
        kmax_[i] = max(ks) if ks else 0
        odd_[i] = bool(np.all(a % 2 == 1))
        cov = R[:, a - 1].all(1)
        if cov.any():
            j = np.argmin(np.where(cov, rsize, 999))
            small_[i] = rsize[j]; smallmask[i] = R[j]
            cover[i] = R[cov].any(0)
        c2 = cov[:35]
        if c2.any():
            small2_[i] = rsize[:35][c2].min()
        for v in xs:
            for ch in str(v):
                digs[i, int(ch)] = True
    n = n_[inv]; lo = lo_[inv]; hi = hi_[inv]; first = first_[inv]; lastx = lastx_[inv]; mono = mono_[inv]
    kmax = kmax_[inv]; odd = odd_[inv]; small = small_[inv]; small2 = small2_[inv]
    tt = np.clip(t, 1, 100) - 1
    in_small = smallmask[inv, tt]
    in_cover = cover[inv, tt]
    span = hi - lo
    inhull = (t >= lo) & (t <= hi)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    ruled = (kmax > 0) | odd
    member = np.where(kmax > 0, t % np.maximum(kmax, 1) == 0, np.where(odd, t % 2 == 1, False))
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    m = (mono != 0) & (dist >= 1) & (dist <= 20)
    fwd = m & (((mono == 1) & (t > hi)) | ((mono == -1) & (t < lo)))
    bwd = m & (((mono == 1) & (t < lo)) | ((mono == -1) & (t > hi)))
    return rate(fwd) - rate(bwd)
