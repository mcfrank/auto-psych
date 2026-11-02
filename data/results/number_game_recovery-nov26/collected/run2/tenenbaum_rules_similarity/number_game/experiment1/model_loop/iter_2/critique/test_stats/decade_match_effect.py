# name: decade_match_effect
# description: Among non-member targets 1-9 away from the nearest example, yes-rate when the target shares its decade (tens digit, 100 counted with 90s) with some example minus when it does not; observed above null means people use decade membership (an unmodelled 'in the 40s'-type concept) more than the model's distance-only intervals, below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    ex_arr = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    codes, inv = np.unique(sets.to_numpy(), return_inverse=True)
    M = np.full((len(codes), 4), -1000)
    for i, s in enumerate(codes):
        e = ex_arr[s]; M[i, :len(e)] = e
    E = M[inv]
    valid = E > -1000
    dist = np.where(valid, np.abs(E - t[:, None]), 10**6).min(axis=1)
    dec = lambda x: np.minimum(x // 10, 9)
    same = (valid & (dec(E) == dec(t)[:, None])).any(axis=1)
    r = df["rating"].to_numpy()
    base = (dist >= 1) & (dist <= 9)
    a, b = base & same, base & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
