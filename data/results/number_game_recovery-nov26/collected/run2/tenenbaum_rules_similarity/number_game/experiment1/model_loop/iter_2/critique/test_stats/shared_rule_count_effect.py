# name: shared_rule_count_effect
# description: Among non-member targets, yes-rate for targets that satisfy >=2 of the simple rules (odd, even, multiples of 3/4/5/6/7/8/9, ends in d) shared by all examples minus yes-rate for targets satisfying exactly 1; observed above null means people demand conjunctive consistency (agreement with all shared regularities) more than the model's single-rule mixture, below null means less.
def test_statistic(df):
    nums = np.arange(1, 101)
    rules = [nums % 2 == 1, nums % 2 == 0] + [nums % m == 0 for m in (3, 4, 5, 6, 7, 8, 9)] + [nums % 10 == d for d in range(10)]
    R = np.array(rules)  # (rules, 100)
    sets = df["number_set"].astype(str).to_numpy()
    codes, inv = np.unique(sets, return_inverse=True)
    C = np.zeros((len(codes), 100), dtype=int)
    ISMEM = np.zeros((len(codes), 100), dtype=bool)
    for i, s in enumerate(codes):
        ex = np.array([int(x) for x in s.split(",")]) - 1
        shared = R[:, ex].all(axis=1)
        C[i] = R[shared].sum(axis=0)
        ISMEM[i, ex] = True
    t = df["target"].astype(int).to_numpy() - 1
    c = C[inv, t]
    nm = ~ISMEM[inv, t]
    r = df["rating"].to_numpy()
    a, b = nm & (c >= 2), nm & (c == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
