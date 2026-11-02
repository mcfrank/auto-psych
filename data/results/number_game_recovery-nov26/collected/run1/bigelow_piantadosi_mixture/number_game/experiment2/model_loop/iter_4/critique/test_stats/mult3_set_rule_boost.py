# name: mult3_set_rule_boost
# description: Sets of >=2 examples that are all multiples of 3 (but not all even): yes-rate for non-example multiple-of-3 targets minus non-example non-multiple-of-3 targets; observed above null_mean means people use the "multiples of 3" rule more than the model predicts, below means the model over-produces it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    ok = {}
    for s in uniq:
        xs = [int(v) for v in s.split(",")]
        ok[s] = len(xs) >= 2 and all(x % 3 == 0 for x in xs) and not all(x % 2 == 0 for x in xs)
    sel = sets.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    isex = (ex == t[:, None]).any(axis=1)
    m = sel & ~isex
    r = df["rating"].values.astype(float)
    a = m & (t % 3 == 0); b = m & (t % 3 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
