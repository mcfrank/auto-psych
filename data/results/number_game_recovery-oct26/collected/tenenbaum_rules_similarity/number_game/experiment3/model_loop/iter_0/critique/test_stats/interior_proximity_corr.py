# name: interior_proximity_corr
# description: Among non-example targets strictly between the min and max of sets of >=2 (span >= 10), the correlation of rating with distance to the nearest example divided by span; observed more negative than null means people's yes falls off with distance from examples inside the range, which uniform-range concepts underproduce.
def test_statistic(df):
    sp = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",")
    t = df["target"].astype(int).values
    dist = np.full(len(df), np.nan)
    lo = np.zeros(len(df)); hi = np.zeros(len(df))
    cache = {}
    for i, (v, tt) in enumerate(zip(sp.values, t)):
        key = (tuple(v), tt)
        if key not in cache:
            ex = np.array([int(x) for x in v])
            span = ex.max() - ex.min()
            if len(ex) >= 2 and span >= 10 and ex.min() < tt < ex.max() and tt not in ex:
                cache[key] = np.abs(ex - tt).min() / span
            else:
                cache[key] = np.nan
        dist[i] = cache[key]
    m = ~np.isnan(dist)
    r = df["rating"].astype(float).values[m]
    d = dist[m]
    if m.sum() < 3 or r.std() == 0 or d.std() == 0:
        return 0.0
    return float(np.corrcoef(d, r)[0, 1])
