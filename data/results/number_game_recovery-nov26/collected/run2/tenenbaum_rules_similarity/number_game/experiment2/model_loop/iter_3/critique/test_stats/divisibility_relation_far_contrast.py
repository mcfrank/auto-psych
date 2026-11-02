# name: divisibility_relation_far_contrast
# description: For non-example targets >=2 that are more than 10 away from every example, yes rate when the target divides or is a multiple of at least one example (pairwise divisibility) minus when it has no such relation; observed above the null means people generalize via pairwise arithmetic relations to single examples beyond the model's whole-set rules (below = less).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    dist = np.array([np.abs(ex[s] - tt).min() for s, tt in zip(sets, t)])
    rel = np.array([bool(np.any((ex[s] % tt == 0) | (tt % ex[s] == 0))) for s, tt in zip(sets, t)])
    ok = (dist > 10) & (t >= 2)
    a, b = ok & rel, ok & ~rel
    y = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
