# name: digit_only_minus_arithmetic_only
# description: For sets consistent with both a digit rule (ends-in-d / contains-d, all examples) and an arithmetic rule (multiples of 3-12, squares, cubes, primes, powers of 2-10), yes rate for non-example targets satisfying only a consistent digit rule minus targets satisfying only a consistent arithmetic rule (e.g. 8,80,48: 28/68 vs 96; 28,98: x8 vs multiples of 7); observed above null means people favour digit rules over equally-consistent arithmetic rules more than the model's size-principle weighting allows.
def test_statistic(df):
    N = np.arange(1, 101)
    digit = [N % 10 == d for d in range(10)] + [np.array([str(d) in str(n) for n in N]) for d in range(10)]
    arith = [N % m == 0 for m in range(3, 13)]
    arith.append(np.isin(N, [n * n for n in range(1, 11)]))
    arith.append(np.isin(N, [n ** 3 for n in range(1, 5)]))
    arith.append(np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in N]))
    for m in range(2, 11):
        arith.append(np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]))
    Dg, Ar = np.array(digit), np.array(arith)
    s_arr = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    dig_only = np.zeros(len(df), bool); ar_only = np.zeros(len(df), bool)
    for s in np.unique(s_arr):
        e = np.array([int(x) for x in s.split(",")])
        dc = Dg[:, e - 1].all(axis=1); ac = Ar[:, e - 1].all(axis=1)
        if not dc.any() or not ac.any():
            continue
        dm = Dg[dc].any(axis=0); am = Ar[ac].any(axis=0)
        idx = (s_arr == s) & ~np.isin(t, e)
        dig_only |= idx & dm[t - 1] & ~am[t - 1]
        ar_only |= idx & am[t - 1] & ~dm[t - 1]
    if dig_only.sum() == 0 or ar_only.sum() == 0:
        return 0.0
    return float(r[dig_only].mean() - r[ar_only].mean())
