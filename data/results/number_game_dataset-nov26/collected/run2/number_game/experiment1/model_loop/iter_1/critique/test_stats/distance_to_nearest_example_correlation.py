# name: distance_to_nearest_example_correlation
# description: Pearson correlation between yes-response and the absolute distance from the target to its nearest example (excluding targets in the set); observed more negative than null means people show a steeper numeric-proximity gradient than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    u = sets.unique()
    arr = {s: np.array([int(x) for x in s.split(",")]) for s in u}
    d = np.empty(len(t))
    sv = sets.values
    for s in u:
        idx = np.where(sv == s)[0]
        d[idx] = np.abs(t[idx][:, None] - arr[s][None, :]).min(axis=1)
    m = d > 0
    if m.sum() < 3 or np.std(d[m]) == 0 or np.std(r[m]) == 0:
        return 0.0
    return float(np.corrcoef(d[m], r[m])[0, 1])
