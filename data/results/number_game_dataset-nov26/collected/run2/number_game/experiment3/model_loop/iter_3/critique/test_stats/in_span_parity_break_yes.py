# name: in_span_parity_break_yes
# description: On multi-example sets of uniform parity, yes rate for opposite-parity targets that lie strictly between the smallest and largest example; observed > null means interval similarity overrides parity for people more than the model allows (observed < null: model's span term leaks yes onto parity violators).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    par = np.full(len(u), -1); lo = np.zeros(len(u)); hi = np.zeros(len(u))
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        ps = {e % 2 for e in ex}
        lo[i], hi[i] = min(ex), max(ex)
        if len(ex) >= 2 and len(ps) == 1:
            par[i] = ps.pop()
    p = par[inv]
    t = df["target"].astype(int).values
    m = (p >= 0) & (t % 2 != p) & (t > lo[inv]) & (t < hi[inv])
    r = df["rating"].astype(float).values
    return float(r[m].mean()) if m.sum() else 0.0
