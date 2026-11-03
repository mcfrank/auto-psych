# name: recency_nearest_example_contrast
# description: Among non-example targets of sets with >=2 examples, yes-rate when the numerically nearest example is the LAST-shown one minus when it is the FIRST-shown one; the model is order-invariant, so observed above model means people weight recently shown examples more (recency), below means primacy.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(dtype=float)
    ss = sets.to_numpy()
    first = np.zeros(len(df), bool)
    last = np.zeros(len(df), bool)
    for s, ex in parsed.items():
        if len(ex) < 2:
            continue
        idx = np.where(ss == s)[0]
        dist = np.abs(t[idx][:, None] - ex[None, :])
        mn = dist.min(axis=1)
        nonmem = mn > 0
        uniq_near = (dist == mn[:, None]).sum(axis=1) == 1
        near = dist.argmin(axis=1)
        ok = nonmem & uniq_near
        first[idx[ok & (near == 0)]] = True
        last[idx[ok & (near == len(ex) - 1)]] = True
    if first.sum() == 0 or last.sum() == 0:
        return 0.0
    return float(r[last].mean() - r[first].mean())
