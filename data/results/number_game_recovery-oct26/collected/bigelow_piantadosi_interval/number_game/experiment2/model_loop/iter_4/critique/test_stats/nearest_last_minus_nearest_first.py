# name: nearest_last_minus_nearest_first
# description: In 3-4 example sets, among non-example targets within 10 of their nearest example, yes rate when that nearest example is the last-presented one minus when it is the first-presented one; observed below the null means the model over-produces recency (people weight the last example less than the volatile model), above means people show even stronger recency.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    ex = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    codes = pd.Categorical(sets, categories=uniq).codes
    mat = np.full((len(uniq), 4), -1000)
    for i, s in enumerate(uniq):
        mat[i, : len(ex[s])] = ex[s]
    E = mat[codes]
    n = (E > 0).sum(axis=1)
    dist = np.abs(E - t[:, None]).astype(float)
    dist[E < 0] = 1e9
    nearest = dist.argmin(axis=1)
    dmin = dist.min(axis=1)
    base = (n >= 3) & (dmin > 0) & (dmin <= 10)
    a = base & (nearest == n - 1)
    b = base & (nearest == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
