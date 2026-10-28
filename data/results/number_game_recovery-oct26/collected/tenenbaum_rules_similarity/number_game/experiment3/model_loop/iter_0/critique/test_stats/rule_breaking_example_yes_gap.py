# name: rule_breaking_example_yes_gap
# description: For sets of >=3, yes-rate to a shown example that violates a simple rule (multiple of k, last digit, powers, squares, primes, parity) shared by all the other examples, minus yes-rate to the other shown examples; observed more negative than null means people discount outlier examples more than the model's 10% noise allows.
def test_statistic(df):
    N = np.arange(1, 101)
    primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97}
    rules = [set(N[N % k == 0]) for k in range(2, 13)]
    rules += [set(N[N % 10 == d]) for d in range(10)]
    rules += [set(N[N % 2 == 1]), primes, {1,4,9,16,25,36,49,64,81,100},
              {1,2,4,8,16,32,64}, {1,3,9,27,81}, {1,4,16,64}, {1,8,27,64}]
    flag = {}
    for s in df["number_set"].astype(str).unique():
        ex = [int(x) for x in s.replace(" ", "").split(",")]
        if len(ex) < 3:
            continue
        for i, e in enumerate(ex):
            others = ex[:i] + ex[i + 1:]
            brk = any(all(o in R for o in others) and e not in R for R in rules)
            flag[(s, e)] = 1 if brk else 0
    keys = list(zip(df["number_set"].astype(str).values, df["target"].astype(int).values))
    f = np.array([flag.get(k, -1) for k in keys])
    r = df["rating"].astype(float).values
    if (f == 1).sum() == 0 or (f == 0).sum() == 0:
        return 0.0
    return float(r[f == 1].mean() - r[f == 0].mean())
