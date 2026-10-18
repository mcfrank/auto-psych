# name: in_band_proximity_gap
# description: Among in-band targets sharing no final digit with any example (not examples), yes rate for targets within 4 of some example minus targets >=10 from every example; t_observed above null_mean means the model under-produces fine-grained numeric-proximity generalization, below means people favour in-band targets far from the examples more than the model (it gives all in-band targets the same magnitude boost).
def test_statistic(df):
    t = df["target"].astype(int).values
    uniq = df["number_set"].astype(str).unique()
    info = {}
    for s in uniq:
        xs = np.array([int(v) for v in s.split(",")])
        info[s] = xs
    sets = df["number_set"].astype(str).values
    mind = np.empty(len(t)); dig = np.zeros(len(t), bool); inband = np.zeros(len(t), bool)
    for s in uniq:
        m = sets == s
        xs = info[s]
        tt = t[m]
        diff = np.abs(tt[:, None] - xs[None, :])
        mind[m] = diff.min(1)
        dig[m] = ((tt[:, None] % 10) == (xs[None, :] % 10)).any(1)
        lo, hi = (xs // 10).min(), (xs // 10).max()
        inband[m] = (tt // 10 >= lo) & (tt // 10 <= hi)
    r = df["rating"].astype(float).values
    base = inband & ~dig & (mind > 0)
    near = base & (mind <= 4)
    far = base & (mind >= 10)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
