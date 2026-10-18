# name: odd_set_parity_generalisation
# description: In multi-example sets whose examples are all odd with at least two distinct final digits, yes-rate on odd targets not matching any example's final digit minus yes-rate on even targets; observed > null means the model under-produces an 'odd numbers' rule (too little parity credit when the shared feature is parity, not digit).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    codes, cats = pd.factorize(sets)
    ok = []
    digs = []
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        ds = set(x % 10 for x in xs)
        digs.append(ds)
        ok.append(len(xs) >= 2 and all(x % 2 == 1 for x in xs) and len(ds) >= 2)
    ok = np.array(ok)
    t = df["target"].astype(int).to_numpy()
    td = t % 10
    match = np.zeros(len(df), dtype=bool)
    for i, ds in enumerate(digs):
        m = codes == i
        match[m] = np.isin(td[m], list(ds))
    inset = ok[codes]
    r = df["rating"].to_numpy().astype(float)
    a = inset & (t % 2 == 1) & ~match
    b = inset & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
