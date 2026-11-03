# name: prime_set_contrast
# description: Among multi-example sets whose examples are all prime, yes-rate for non-example odd prime targets minus yes-rate for non-example odd composite targets; observed above model means people generalise the primes concept more sharply than the model's family-weighted prime prior predicts, below means they treat primes like generic odd numbers more.
def test_statistic(df):
    primes = {n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1))}
    sets = df["number_set"].astype(str)
    keep, exsets = [], {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        if len(ex) >= 2 and all(e in primes for e in ex):
            keep.append(s)
            exsets[s] = set(ex)
    if not keep:
        return 0.0
    sub = df[sets.isin(keep).to_numpy()]
    t = sub["target"].astype(int).to_numpy()
    ss = sub["number_set"].astype(str).to_numpy()
    inset = np.array([x in exsets[s] for s, x in zip(ss, t)])
    r = sub["rating"].to_numpy(dtype=float)
    isp = np.isin(t, list(primes))
    odd = t % 2 == 1
    a = r[isp & odd & ~inset]
    b = r[~isp & odd & (t > 1) & ~inset]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
