# name: round_number_target_bonus
# description: In sets whose examples are not all multiples of 10, yes-rate for targets that are multiples of 10 minus other targets of the same parity as the set (even sets only); observed > null means people favour round-number targets a priori (a bias the model, which only credits shared properties, lacks), observed < null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        ok[s] = all(e % 2 == 0 for e in ex) and not all(e % 10 == 0 for e in ex)
    o = sets.map(ok).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    r = o & (t % 10 == 0)
    nr = o & (t % 2 == 0) & (t % 10 != 0)
    if r.sum() == 0 or nr.sum() == 0:
        return 0.0
    return float(y[r].mean() - y[nr].mean())
