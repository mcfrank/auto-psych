# name: narrow_rule_outside_distance_slope
# description: For multi-example rule sets (multiple-of-k or odd), least-squares slope of yes vs distance (per 10 units) for rule members outside the examples' span; observed more negative than null means endorsement falls off with distance faster than the model's range-limited rules predict.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = (df["rating"].astype(float).to_numpy() > 0.5).astype(float)
    uniq = pd.unique(s)
    info = {}
    for u in uniq:
        xs = [int(v) for v in str(u).split(",") if v.strip() != ""]
        ks = [k for k in range(2, 11) if all(x % k == 0 for x in xs)]
        odd = all(x % 2 == 1 for x in xs)
        r10 = len(set(x % 10 for x in xs)) == 1 and len(xs) > 1
        info[u] = (len(xs), min(xs), max(xs), max(ks) if ks else 0, odd, r10, xs[0] % 10, xs)
    n = s.map(lambda u: info[u][0]).to_numpy()
    lo = s.map(lambda u: info[u][1]).to_numpy()
    hi = s.map(lambda u: info[u][2]).to_numpy()
    kmax = s.map(lambda u: info[u][3]).to_numpy()
    odd = s.map(lambda u: info[u][4]).to_numpy().astype(bool)
    r10 = s.map(lambda u: info[u][5]).to_numpy().astype(bool)
    last = s.map(lambda u: info[u][6]).to_numpy()
    span = hi - lo
    inhull = (t >= lo) & (t <= hi)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    ruled = (kmax > 0) | odd
    member = np.where(kmax > 0, t % np.maximum(kmax, 1) == 0, np.where(odd, t % 2 == 1, False))
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    m = ruled & (n > 1) & member & (dist > 0)
    if m.sum() < 3:
        return 0.0
    d = dist[m] / 10.0
    yy = y[m]
    v = np.var(d)
    return float(np.mean((d - d.mean()) * (yy - yy.mean())) / v) if v > 0 else 0.0
