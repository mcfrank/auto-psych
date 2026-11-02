# name: first_vs_last_example_proximity
# description: For 2+ example sets, yes rate on non-example targets within 10 of the first-shown example (but not the last) minus targets within 10 of the last-shown example (but not the first); the model is order-invariant, so observed > null means a primacy advantage and observed < null a recency advantage.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    idx = (n - 1).clip(lower=0).astype(int).to_numpy()
    last = pd.Series(ex.to_numpy()[np.arange(len(ex)), idx], index=df.index)
    first = ex[0]
    inset = ex.eq(t, axis=0).any(axis=1)
    nf = (t - first).abs() <= 10
    nl = (t - last).abs() <= 10
    base = (n >= 2) & ~inset
    r = df["rating"].astype(float)
    a = base & nf & ~nl
    b = base & nl & ~nf
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
