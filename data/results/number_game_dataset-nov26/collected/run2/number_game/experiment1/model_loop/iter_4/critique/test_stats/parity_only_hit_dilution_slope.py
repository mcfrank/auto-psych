# name: parity_only_hit_dilution_slope
# description: For parity-homogeneous sets, among targets that match the shared parity but none of the set's other shared multiple-of-m (3..12) or final-digit properties, the yes-rate in sets with >=2 such extra shared properties minus sets with 0; the model's normalized overlap predicts strong dilution (negative), so observed > null means people treat parity as sufficient regardless of how many other properties the examples share, observed < null means dilution is even stronger.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        ms = [m for m in range(3, 13) if all(e % m == 0 for e in ex)]
        d = {e % 10 for e in ex}
        fd = d.pop() if (len(ex) >= 2 and len(d) == 1) else -1
        pr = {e % 2 for e in ex}
        par = pr.pop() if len(pr) == 1 else -1
        info[s] = (par, ms, fd)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n_extra = np.zeros(len(df)); keep = np.zeros(len(df), dtype=bool)
    sv = sets.to_numpy()
    for s, (par, ms, fd) in info.items():
        idx = sv == s
        if par < 0:
            continue
        tt = t[idx]
        k = (tt % 2 == par)
        for m in ms:
            k &= (tt % m != 0)
        if fd >= 0:
            k &= (tt % 10 != fd)
        keep[idx] = k
        n_extra[idx] = len(ms) + (1 if fd >= 0 else 0)
    a = keep & (n_extra >= 2)
    b = keep & (n_extra == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
