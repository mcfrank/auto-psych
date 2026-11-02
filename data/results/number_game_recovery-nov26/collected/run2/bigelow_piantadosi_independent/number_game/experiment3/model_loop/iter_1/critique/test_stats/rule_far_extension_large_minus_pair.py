# name: rule_far_extension_large_minus_pair
# description: For sets whose examples all share a multiples-of-m rule (m 3-12, or m=2 with no larger m) or a common final digit, yes-rate on rule-consistent non-example targets 10+ outside the span for 3-4-example sets minus 2-example sets; observed above null_mean means rules overtake ranges with more examples faster than the model predicts (below = slower).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    info = {}
    for s in u:
        a = np.array([int(x) for x in s.split(",")])
        ms = [m for m in range(3, 13) if np.all(a % m == 0)]
        if not ms and np.all(a % 2 == 0):
            ms = [2]
        fd = int(a[0] % 10) if np.all(a % 10 == a[0] % 10) else -1
        info[s] = (ms, fd, a.min(), a.max(), len(a))
    ns = df["number_set"].to_numpy()
    match = np.zeros(len(df), bool); lo = np.zeros(len(df)); hi = np.zeros(len(df)); n = np.zeros(len(df))
    for s, (ms, fd, mn, mx, k) in info.items():
        idx = ns == s
        tt = t[idx]
        mm = np.zeros(len(tt), bool)
        for m in ms:
            mm |= tt % m == 0
        if fd >= 0:
            mm |= tt % 10 == fd
        match[idx] = mm; lo[idx] = mn; hi[idx] = mx; n[idx] = k
    far = (t <= lo - 10) | (t >= hi + 10)
    r = df["rating"].astype(float).to_numpy()
    a = match & far & (n >= 3); b = match & far & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
