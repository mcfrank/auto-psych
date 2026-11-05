# name: power_of_two_member_gap
# description: On multi-example sets made only of powers of two (e.g. 8,16 / 16,32,2,8), yes rate for power-of-two targets minus yes rate for other targets; observed > null means people pick out the powers-of-two rule more sharply than the model (which spreads yes over even/multiple targets).
def test_statistic(df):
    pw = {2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ok = np.zeros(len(u), bool)
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        ok[i] = len(ex) >= 2 and all(e in pw for e in ex)
    m = ok[inv]
    t = df["target"].astype(int).values
    hit = np.isin(t, list(pw))
    r = df["rating"].astype(float).values
    a, b = m & hit, m & ~hit
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
