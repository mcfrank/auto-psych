# name: last_vs_first_example_proximity
# description: Among multi-example sets, yes rate for targets within 5 of the last-shown example (and farther than 5 from the first) minus yes rate for targets within 5 of the first-shown example (and farther than 5 from the last); the model is order-invariant (null ~0), so observed above null_mean means a recency advantage, below a primacy advantage.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1).to_numpy()
    arr = s.to_numpy(dtype=float)
    first = arr[:, 0]
    last = arr[np.arange(len(arr)), np.maximum(n - 1, 0)]
    t = df["target"].astype(int).to_numpy().astype(float)
    df_ = np.abs(t - first); dl = np.abs(t - last)
    y = df["rating"].astype(float).to_numpy()
    a = (n >= 2) & (dl <= 5) & (df_ > 5)
    b = (n >= 2) & (df_ <= 5) & (dl > 5)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
