# name: prime_majority_exception_contrast
# description: For sets of >=3 examples where all but one example is prime and the exception is odd (e.g. 17,11,65,53; 99,59,31,67), yes rate on non-example prime targets minus odd composite targets; observed > null means people keep a "primes" hypothesis despite the exception more than the model (which only has strays), observed < null means they abandon it for "odd" more.
def test_statistic(df):
    def isp(n):
        return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ok = set()
    for s in pd.unique(sets):
        e = [int(x) for x in s.split(",")]
        np_ = [x for x in e if not isp(x)]
        if len(e) >= 3 and len(np_) == 1 and np_[0] % 2 == 1:
            ok.add(s)
    m = np.isin(sets, list(ok))
    exm = np.array([str(x) in s.split(",") for x, s in zip(t, sets)]) if m.any() else np.zeros(len(t), bool)
    pr = np.isin(t, [n for n in range(1, 101) if isp(n)])
    a = r[m & ~exm & pr]
    b = r[m & ~exm & ~pr & (t % 2 == 1) & (t > 1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
