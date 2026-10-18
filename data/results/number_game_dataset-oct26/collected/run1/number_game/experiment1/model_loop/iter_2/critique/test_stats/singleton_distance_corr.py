# name: singleton_distance_corr
# description: For 1-example sets only, Pearson correlation of rating with |target - example|; observed more negative than null means the model under-produces magnitude-based similarity gradients around a single example.

def _prep(df):
    # Per-row: n examples, min distance to examples, range, shared-property mask, target mask.
    primes = [n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))]
    mask = np.zeros(101, dtype=np.int64)
    for n in range(1, 101):
        b = 0
        if n % 2 == 0: b |= 1
        else: b |= 2
        if n % 3 == 0: b |= 4
        if n % 4 == 0: b |= 8
        if n % 5 == 0: b |= 16
        if n % 10 == 0: b |= 32
        if n in primes: b |= 64
        if n in (1, 4, 9, 16, 25, 36, 49, 64, 81, 100): b |= 128
        if n in (1, 2, 4, 8, 16, 32, 64): b |= 256
        b |= 1 << (9 + n % 10)
        mask[n] = b
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    info = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        sh = -1
        for e in ex:
            sh &= int(mask[e])
        info[s] = (len(ex), min(ex), max(ex), sh, ex)
    t = df["target"].astype(int).to_numpy()
    nex = sets.map(lambda s: info[s][0]).to_numpy()
    lo = sets.map(lambda s: info[s][1]).to_numpy()
    hi = sets.map(lambda s: info[s][2]).to_numpy()
    sh = sets.map(lambda s: info[s][3]).to_numpy().astype(np.int64)
    exm = np.full((len(df), 4), np.nan)
    for s in uniq:
        idx = (sets == s).to_numpy()
        ex = info[s][4]
        exm[idx, :len(ex)] = ex
    dmin = np.nanmin(np.abs(exm - t[:, None]), axis=1)
    tm = mask[t]
    r = df["rating"].astype(float).to_numpy()
    return t, nex, lo, hi, sh, tm, dmin, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    t, nex, lo, hi, sh, tm, dmin, r = _prep(df)
    m = (nex == 1) & (dmin > 0)
    if m.sum() < 3 or np.std(r[m]) == 0 or np.std(dmin[m]) == 0:
        return 0.0
    return float(np.corrcoef(r[m], dmin[m])[0, 1])
