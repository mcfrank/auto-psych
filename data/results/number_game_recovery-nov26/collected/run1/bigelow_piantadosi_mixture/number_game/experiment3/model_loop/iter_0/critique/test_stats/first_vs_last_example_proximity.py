# name: first_vs_last_example_proximity
# description: In sets of >=3 examples, yes rate for non-example targets within 5 of the first example (and >10 from the last) minus yes rate for targets within 5 of the last example (and >10 from the first); observed < model means the model over-produces primacy (first-example-anchored guessing) relative to people.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tg = df["target"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    first = ex[:, 0]
    idx = np.clip(n - 1, 0, ex.shape[1] - 1)
    last = ex[np.arange(len(ex)), idx]
    inset = (np.where(valid, ex, -1) == tg[:, None]).any(axis=1)
    df1 = np.abs(tg - first)
    dl = np.abs(tg - last)
    base = (n >= 3) & ~inset
    a = base & (df1 <= 5) & (dl > 10)
    b = base & (dl <= 5) & (df1 > 10)
    r = df["rating"].astype(float).to_numpy()
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
