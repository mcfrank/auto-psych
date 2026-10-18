# name: cross_position_digit_overlap
# description: Among targets matching no example's final digit and whose decade lies outside the examples' decade range, yes-rate when a target digit appears in an example in the other position (target tens = an example's units, or target units = an example's tens) minus yes-rate otherwise; observed > null means people credit position-free digit identity (e.g. 31 vs 13) that the model, using only final digits and decade bands, misses.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        units = {x % 10 for x in xs}
        tens = {(x // 10) % 10 for x in xs if x >= 10}
        info[s] = (units, tens, min(x // 10 for x in xs), max(x // 10 for x in xs))
    sv = sets.values
    t = df["target"].astype(int).values
    tu = t % 10
    tt = (t // 10) % 10
    elig = np.zeros(len(t), bool)
    cross = np.zeros(len(t), bool)
    for i, (s, ti) in enumerate(zip(sv, t)):
        units, tens, lo, hi = info[s]
        td = ti // 10
        if (tu[i] in units) or (lo <= td <= hi):
            continue
        elig[i] = True
        cross[i] = (ti >= 10 and tt[i] in units) or (tu[i] in tens)
    r = df["rating"].values.astype(float)
    a = elig & cross
    b = elig & ~cross
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
