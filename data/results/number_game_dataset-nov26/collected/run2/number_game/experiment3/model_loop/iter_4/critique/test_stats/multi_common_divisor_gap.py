# name: multi_common_divisor_gap
# description: On 2-4-example sets, yes rate for non-member targets >=2 that divide every example (e.g. 15 for 90,45) minus yes rate for other non-member targets on those same kinds of sets; observed > null means people generalize downward to common divisors more than the model (whose properties only look at what the examples are multiples of), observed < null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ex = [[int(x) for x in s.split(",")] for s in u]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    T = np.arange(101)
    divm = np.zeros((len(u), 101), bool)
    mem = np.zeros((len(u), 101), bool)
    multi = np.zeros(len(u), bool)
    for i, e in enumerate(ex):
        multi[i] = len(e) >= 2
        dv = np.ones(101, bool)
        for x in e:
            dv &= (T >= 2) & (x % np.maximum(T, 1) == 0)
            mem[i, x] = True
        divm[i] = dv
    isdiv = divm[inv, t]
    ism = mem[inv, t]
    m = multi[inv] & ~ism
    a, b = m & isdiv, m & ~isdiv
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
