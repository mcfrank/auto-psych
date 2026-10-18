# name: ordered_vs_unordered_examples
# description: Among sets with >=3 examples, overall yes-rate on non-example targets for sets shown in monotonic (ascending or descending) order minus for sets shown in non-monotonic order; the model is order-invariant, so observed far from null means presentation order itself (a sequence-like display) changes how broadly people generalise (observed > null: ordered sets generalise more broadly than modelled).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        if len(xs) < 3:
            info[s] = (-1, set(xs))
            continue
        dif = np.diff(xs)
        mono = bool(np.all(dif > 0) or np.all(dif < 0))
        info[s] = (1 if mono else 0, set(xs))
    sv = sets.values
    t = df["target"].astype(int).values
    k = np.array([info[s][0] for s in sv])
    is_ex = np.array([ti in info[s][1] for s, ti in zip(sv, t)])
    r = df["rating"].values.astype(float)
    a = (k == 1) & ~is_ex
    b = (k == 0) & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
