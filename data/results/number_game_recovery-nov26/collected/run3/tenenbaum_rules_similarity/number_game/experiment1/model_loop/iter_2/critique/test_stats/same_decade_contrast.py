# name: same_decade_contrast
# description: Among non-example targets whose distance to the nearest example is 2-9, yes-rate when the target shares its tens digit with some example minus when it shares no example's tens digit; observed above model means people use decade (tens-digit) similarity beyond what the model's rules and intervals produce, below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(dtype=float)
    ss = sets.to_numpy()
    dec = np.zeros(len(df), bool)
    mind = np.zeros(len(df))
    for s, ex in parsed.items():
        idx = np.where(ss == s)[0]
        tt = t[idx]
        mind[idx] = np.abs(tt[:, None] - ex[None, :]).min(axis=1)
        dec[idx] = ((tt[:, None] // 10) == (ex[None, :] // 10)).any(axis=1)
    sel = (mind >= 2) & (mind <= 9)
    a = r[sel & dec]
    b = r[sel & ~dec]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
