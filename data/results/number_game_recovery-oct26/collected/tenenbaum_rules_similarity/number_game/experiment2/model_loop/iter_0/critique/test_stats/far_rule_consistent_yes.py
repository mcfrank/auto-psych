# name: far_rule_consistent_yes
# description: Among multi-example sets with all examples of one parity and span <= 20, yes-rate to same-parity non-example targets >= 25 from the nearest example; observed above null means the model's range restriction over-suppresses far rule members (below null: under-suppresses).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = sets.apply(lambda l: np.array([int(x) for x in l]))
    t = df["target"].astype(int).values
    n = ex.apply(len).values
    mn = ex.apply(np.min).values; mx = ex.apply(np.max).values
    par = ex.apply(lambda a: int(a[0] % 2) if np.all(a % 2 == a[0] % 2) else -1).values
    dist = np.array([np.min(np.abs(a - x)) for a, x in zip(ex.values, t)])
    mask = (n >= 2) & (par >= 0) & (mx - mn <= 20) & (t % 2 == par) & (dist >= 25)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
