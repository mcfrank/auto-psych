# name: parity_only_credit_odd_minus_even_sets
# description: Among sets of >=2 examples that share one parity but do NOT all share a final digit, yes rate for same-parity targets matching no example's final digit (not examples), all-odd sets minus all-even sets; t_observed above null_mean means people give odd-number sets more parity generalization than even sets (the model treats parity symmetrically), below means even sets get more.
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        xs = [int(x) for x in s.split(",")]
        par = set(x % 2 for x in xs)
        dig = set(x % 10 for x in xs)
        ok = len(xs) >= 2 and len(par) == 1 and len(dig) > 1
        info[s] = (ok, xs[0] % 2, dig, set(xs))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    sel = np.array([info[s][0] and v % 2 == info[s][1] and (v % 10 not in info[s][2]) and (v not in info[s][3]) for s, v in zip(sets, t)])
    odd = np.array([info[s][1] == 1 for s in sets])
    a = sel & odd
    b = sel & ~odd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
