# name: fine_numeric_proximity_gap
# description: Among targets matching no example's final digit, yes-rate for targets within 5 of the nearest example minus yes-rate for targets more than 20 away; observed > null means the model under-produces fine-grained numeric-proximity generalisation (beyond its decade-band term).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    t = df["target"].astype(int).to_numpy()
    mind = np.full(len(df), 1000.0)
    match = np.zeros(len(df), dtype=bool)
    codes, cats = pd.factorize(sets)
    for i, s in enumerate(cats):
        xs = np.array([int(x) for x in s.split(",")])
        m = codes == i
        tt = t[m]
        mind[m] = np.abs(tt[:, None] - xs[None, :]).min(axis=1)
        match[m] = np.isin(tt % 10, xs % 10)
    r = df["rating"].to_numpy().astype(float)
    near = (~match) & (mind <= 5) & (mind > 0)
    far = (~match) & (mind > 20)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
