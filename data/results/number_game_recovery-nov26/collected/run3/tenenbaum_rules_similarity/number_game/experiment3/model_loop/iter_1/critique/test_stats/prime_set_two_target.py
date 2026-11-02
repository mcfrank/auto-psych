# name: prime_set_two_target
# description: Yes rate for target 2 minus yes rate for other prime targets, among sets whose examples (>=2) are all odd primes; observed below null means people reject the even prime 2 more than the prime-rule learner predicts.
def test_statistic(df):
    P = {n for n in range(2, 101) if all(n % a for a in range(2, int(n**0.5) + 1))}
    sets = df["number_set"].astype(str)
    u = sets.unique()
    ok = {x: (len(x.split(",")) >= 2 and all(int(v) in P and int(v) != 2 for v in x.split(","))) for x in u}
    m = sets.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    r = df["rating"].values
    two = m & (t == 2)
    oth = m & np.isin(t, list(P - {2}))
    if not two.any() or not oth.any():
        return 0.0
    return float(r[two].mean() - r[oth].mean())
