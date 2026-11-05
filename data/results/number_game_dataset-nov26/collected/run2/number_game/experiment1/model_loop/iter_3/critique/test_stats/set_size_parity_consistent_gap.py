# name: set_size_parity_consistent_gap
# description: Among parity-homogeneous sets, yes-rate for parity-consistent targets in 3-4-example sets minus in 1-2-example sets; observed < null means people narrow generalization with more examples (a size-principle effect the model lacks), observed > null means more examples make people more permissive than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    par = ex.map(lambda l: (lambda p: p[0] if len(set(p)) == 1 else -1)([int(x) % 2 for x in l])).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    cons = (par >= 0) & ((t % 2) == par)
    a = cons & (n >= 3)
    b = cons & (n <= 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
