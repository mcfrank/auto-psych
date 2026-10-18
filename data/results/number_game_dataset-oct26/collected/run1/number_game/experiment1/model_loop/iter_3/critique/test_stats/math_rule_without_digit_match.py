# name: math_rule_without_digit_match
# description: Among targets sharing no final digit with any example on 2-4 example sets, yes rate for targets that share every non-parity arithmetic property common to all examples (mult of 3/4/5/10, power of 2, perfect square; set must have >=1 such property) minus yes rate for targets lacking all of them; observed above null means people generalize by mathematical rules that the final-digit model cannot see.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    exf = np.where(valid, ex, 1.0)
    p2 = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
    sq = np.arange(1, 11, dtype=float) ** 2
    def props(x):
        return [x % 3 == 0, x % 4 == 0, x % 5 == 0, x % 10 == 0, np.isin(x, p2), np.isin(x, sq)]
    pe = props(exf)
    pt = props(t)
    shared = [np.all(np.where(valid, p, True), axis=1) for p in pe]
    nshared = np.sum(shared, axis=0)
    has_all = np.ones(len(t), bool)
    has_none = np.ones(len(t), bool)
    for s, q in zip(shared, pt):
        has_all &= (~s) | q
        has_none &= (~s) | (~q)
    digit_match = np.any(valid & ((exf % 10) == (t[:, None] % 10)), axis=1)
    base = (n >= 2) & (nshared >= 1) & (~digit_match)
    r = df["rating"].to_numpy().astype(float)
    a = base & has_all
    b = base & has_none
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
