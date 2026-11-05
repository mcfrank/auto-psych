# name: contained_digit_only_effect
# description: In sets whose examples all contain a digit d but share no final digit (e.g. 20,32,92 share '2'), among targets of the examples' shared parity (if any), yes-rate for targets containing d minus those not containing d; observed < null means the model over-weights the shared contained-digit feature (people largely ignore it), observed > null means it under-weights it.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    ts = t.astype(str)
    a = np.zeros(len(t), bool); b = np.zeros(len(t), bool)
    for s in pd.unique(sets):
        xs = [int(x) for x in s.split(",")]
        if len(xs) < 2 or len({x % 10 for x in xs}) == 1:
            continue
        digs = [d for d in "0123456789" if all(d in str(x) for x in xs)]
        if not digs:
            continue
        idx = sets == s
        par = {x % 2 for x in xs}
        if len(par) == 1:
            idx = idx & (t % 2 == list(par)[0])
        has = np.zeros(len(t), bool)
        for d in digs:
            has |= np.char.find(ts, d) >= 0
        a |= idx & has
        b |= idx & ~has
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
