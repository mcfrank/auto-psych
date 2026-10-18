# name: rule_consistent_minus_inconsistent
# description: Among multi-example sets whose examples all share at least one arithmetic property (even/odd, mult of 3/4/5/10, same final digit, prime, square, power of 2), yes rate for targets having ALL shared properties minus yes rate for targets lacking any; observed above null means people apply the shared rule more sharply than graded summed similarity predicts.
def test_statistic(df):
    sq = {k * k for k in range(1, 11)}
    p2 = {2 ** k for k in range(0, 7)}
    pr = {m for m in range(2, 101) if all(m % k for k in range(2, int(m ** 0.5) + 1))}
    def props(x):
        return np.array([x % 2 == 0, x % 2 == 1, x % 3 == 0, x % 4 == 0, x % 5 == 0,
                         x % 10 == 0, x in pr, x in sq, x in p2], dtype=bool)
    tprops = np.array([props(t) for t in range(101)])
    targets = df["target"].astype(int).to_numpy()
    cons = np.full(len(df), np.nan)
    codes, uniq = pd.factorize(df["number_set"])
    for k, s in enumerate(uniq):
        xs = [int(v) for v in s.split(",")]
        if len(xs) < 2:
            continue
        shared = np.all(np.array([props(x) for x in xs]), axis=0)
        same_last = len({x % 10 for x in xs}) == 1
        if not shared.any() and not same_last:
            continue
        idx = np.where(codes == k)[0]
        tt = targets[idx]
        ok = np.all(tprops[tt][:, shared], axis=1) if shared.any() else np.ones(len(tt), bool)
        if same_last:
            ok = ok & (tt % 10 == xs[0] % 10)
        cons[idx] = ok.astype(float)
    r = df["rating"].astype(float).to_numpy()
    m1 = cons == 1
    m0 = cons == 0
    if m1.sum() == 0 or m0.sum() == 0:
        return 0.0
    return float(r[m1].mean() - r[m0].mean())
