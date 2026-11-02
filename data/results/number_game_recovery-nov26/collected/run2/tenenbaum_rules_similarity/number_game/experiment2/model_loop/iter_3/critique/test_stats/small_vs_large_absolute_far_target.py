# name: small_vs_large_absolute_far_target
# description: For targets more than 15 away from every example, yes rate for targets 1-20 minus targets 81-100 (absolute magnitude, not relative to examples); observed above the null means people favour small numbers as generic program outputs beyond what the model's concepts and uniform noise produce (below = favour large numbers).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    dist = np.array([np.abs(ex[s] - tt).min() for s, tt in zip(sets, t)])
    ok = dist > 15
    a, b = ok & (t <= 20), ok & (t >= 81)
    y = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
