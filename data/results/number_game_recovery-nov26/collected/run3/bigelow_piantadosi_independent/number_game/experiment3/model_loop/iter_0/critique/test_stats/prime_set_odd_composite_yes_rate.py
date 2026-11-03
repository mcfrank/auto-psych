# name: prime_set_odd_composite_yes_rate
# description: After sets of 2+ examples that are all odd primes (not all ending in the same digit), yes rate to odd composite targets more than 5 from every example; observed above null means the model's prime-over-odd shift over-suppresses the "odd numbers" reading, below means it under-suppresses it.
def test_statistic(df):
    pr = np.zeros(102, dtype=bool)
    for k in range(2, 101):
        pr[k] = all(k % j for j in range(2, int(k ** 0.5) + 1))
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").reindex(columns=range(4)).values
    ok = ~np.isnan(ex)
    exi = np.where(ok, ex, 0).astype(int)
    n = ok.sum(1)
    allp = np.all(~ok | (pr[exi] & (exi % 2 == 1)), axis=1)
    last = np.where(ok, exi % 10, -1)
    same_last = np.all((last == last[:, [0]]) | ~ok, axis=1)
    t = df["target"].astype(int).values
    dist = np.min(np.where(ok, np.abs(exi - t[:, None]), 1000), axis=1)
    m = (n >= 2) & allp & ~same_last & (t % 2 == 1) & ~pr[t] & (t > 1) & (dist > 5)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
