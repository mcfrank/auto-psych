# name: smallest_rule_member_setsize_contrast
# description: For multi-example sets whose smallest consistent arithmetic rule (odd/even, primes, squares, cubes, multiples of 3-12, powers of 2-10, ends in d; <=33 members) contains all examples, yes rate on non-example members of that rule for 4-example sets minus 2-example sets; observed < null means people sharpen toward the rule with more examples less than the size principle predicts, observed > null means more.
def test_statistic(df):
    N = range(1, 101)
    rules = [set(n for n in N if n % 2 == 1), set(range(2, 101, 2)),
             set(n for n in N if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))),
             set(k * k for k in range(1, 11)), set(k ** 3 for k in range(1, 5))]
    rules += [set(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [set(m ** k for k in range(1, 8) if m ** k <= 100) for m in range(2, 11)]
    rules += [set(n for n in N if n % 10 == d) for d in range(10)]
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    member = np.zeros(len(df), dtype=bool)
    nex = np.zeros(len(df), dtype=int)
    for s in pd.unique(sets):
        e = set(int(x) for x in s.split(","))
        if len(e) < 2:
            continue
        cons = [R for R in rules if e <= R and len(R) <= 33]
        if not cons:
            continue
        R = min(cons, key=len)
        m = sets == s
        mem = np.isin(t, list(R - e))
        member |= m & mem
        nex[m] = len(e)
    a = r[member & (nex == 4)]
    b = r[member & (nex == 2)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
