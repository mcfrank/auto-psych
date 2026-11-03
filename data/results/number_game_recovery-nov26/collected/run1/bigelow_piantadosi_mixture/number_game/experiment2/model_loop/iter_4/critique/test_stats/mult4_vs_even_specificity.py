# name: mult4_vs_even_specificity
# description: Sets of >=2 examples that are all multiples of 4: yes-rate for non-example multiple-of-4 targets minus non-example even-but-not-multiple-of-4 targets; observed above null_mean means people pick the more specific "multiples of 4" rule over "even" more sharply than the model (size principle too weak), below means the model is too specific.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        ok[s] = len(xs) >= 2 and all(x % 4 == 0 for x in xs)
    sel = sets.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    isex = (ex == t[:, None]).any(axis=1)
    m = sel & ~isex
    r = df["rating"].values.astype(float)
    a = m & (t % 4 == 0); b = m & (t % 2 == 0) & (t % 4 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
