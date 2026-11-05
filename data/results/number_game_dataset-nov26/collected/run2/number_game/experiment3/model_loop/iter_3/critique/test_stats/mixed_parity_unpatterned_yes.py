# name: mixed_parity_unpatterned_yes
# description: Yes rate on multi-example sets with mixed parity and no shared final digit (unpatterned sets like 92,82,27,42 or 48,99,3), over all their targets; observed < null means the model's flat doubt-mode yes is too generous for sets with no shared rule (observed > null: too stingy).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ok = np.zeros(len(u), bool)
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        ok[i] = len(ex) >= 2 and len({e % 2 for e in ex}) == 2 and len({e % 10 for e in ex}) > 1
    m = ok[inv]
    r = df["rating"].astype(float).values
    return float(r[m].mean()) if m.sum() else 0.0
