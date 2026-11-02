# name: last_vs_first_example_proximity
# description: Among sets of 2+ examples and non-example targets, yes rate for targets within 10 of the last-shown example (and more than 10 from the first) minus yes rate for targets within 10 of the first-shown example (and more than 10 from the last); the model ignores presentation order, so observed above the null means a recency effect in generalization (below = primacy).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    parts = sets.str.split(",")
    first = parts.str[0].astype(int).to_numpy()
    last = parts.str[-1].astype(int).to_numpy()
    n = parts.str.len().to_numpy()
    t = df["target"].astype(int).to_numpy()
    exm = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    inset = (exm == df["target"].astype(int).to_numpy()[:, None]).any(axis=1)
    df1 = np.abs(t - first); dl = np.abs(t - last)
    base = (n >= 2) & ~inset
    a = base & (dl <= 10) & (df1 > 10)
    b = base & (df1 <= 10) & (dl > 10)
    r = df["rating"].to_numpy()
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
