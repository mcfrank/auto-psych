# name: last_vs_first_example_proximity
# description: In multi-example sets, yes rate for targets within 5 of the last-shown example (but not within 5 of the first) minus that for targets within 5 of the first-shown example (but not the last), excluding examples; the model is order-invariant, so observed above the null means humans weight the most recent example more (recency), below means primacy.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(df)), n - 1]
    inset = np.any(ex == t[:, None], axis=1)
    nf = np.abs(t - first) <= 5; nl = np.abs(t - last) <= 5
    base = (n >= 2) & ~inset
    a = y[base & nl & ~nf]; b = y[base & nf & ~nl]
    return float((a.mean() if a.size else 0.0) - (b.mean() if b.size else 0.0))
