# name: power_sequence_concept
# description: Among sets of 2+ examples that are all powers of 2 or all perfect squares, yes-rate on non-example targets in that same family minus non-example targets outside it; observed > null means people infer "powers of 2" / "square numbers" concepts that the model (only multiples-of-g, final digit and factor similarity) under-produces.
def test_statistic(df):
    p2 = [1, 2, 4, 8, 16, 32, 64]
    sq = [i * i for i in range(1, 11)]
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    k = ex.notna().sum(axis=1).values
    allp2 = (ex.isin(p2) | ex.isna()).all(axis=1).values & (k >= 2)
    allsq = (ex.isin(sq) | ex.isna()).all(axis=1).values & (k >= 2) & ~allp2
    isex = (ex.values == t.values[:, None]).any(axis=1)
    infam = np.where(allp2, t.isin(p2).values, t.isin(sq).values)
    s = (allp2 | allsq) & ~isex
    r = df["rating"].values.astype(float)
    a = s & infam; b = s & ~infam
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
