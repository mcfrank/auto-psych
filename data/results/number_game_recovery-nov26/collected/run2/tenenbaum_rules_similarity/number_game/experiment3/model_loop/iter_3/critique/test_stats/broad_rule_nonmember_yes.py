# name: broad_rule_nonmember_yes
# description: Among sets of 2+ distinct examples whose smallest consistent rule has 21-50 members (e.g. odd, even, multiples of 3 or 4, primes), yes rate on non-example targets outside that rule; observed below null means people respect broad rules (parity, multiples of 3) more strictly than the model, above null means people ignore them more.
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [N % 2 == 1, N % 2 == 0, np.isin(N, [k * k for k in range(1, 11)]), np.isin(N, [1, 8, 27, 64]),
             np.isin(N, [2, 4, 8, 16, 32, 64]), np.isin(N, [3, 9, 27, 81]), np.isin(N, [4, 16, 64]),
             np.isin(N, [5, 25]), np.isin(N, [6, 36]), np.isin(N, [7, 49]), np.isin(N, [8, 64]), np.isin(N, [9, 81]),
             np.isin(N, [n for n in N if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))])]
    rules += [N % m == 0 for m in range(3, 13)] + [N % 10 == d for d in range(10)]
    M = np.array(rules)
    size = M.sum(1)
    member = {}
    for s in df["number_set"].unique():
        ex = sorted(set(int(x) for x in str(s).split(",")))
        if len(ex) < 2:
            continue
        ok = M[:, np.array(ex) - 1].all(1)
        if not ok.any():
            continue
        r = np.where(ok)[0][np.argmin(size[ok])]
        if 20 < size[r] <= 50:
            member[s] = (M[r], ex[0], ex[-1], set(ex))
    if not member:
        return 0.0
    sub = df[df["number_set"].isin(list(member))]
    t = sub["target"].astype(int).to_numpy()
    sets = sub["number_set"].to_numpy()
    mem = np.array([member[s][0][ti - 1] for s, ti in zip(sets, t)])
    lo = np.array([member[s][1] for s in sets]); hi = np.array([member[s][2] for s in sets])
    isex = np.array([ti in member[s][3] for s, ti in zip(sets, t)])
    m = (~mem) & (~isex)
    return float(sub["rating"].to_numpy()[m].mean()) if m.any() else 0.0
