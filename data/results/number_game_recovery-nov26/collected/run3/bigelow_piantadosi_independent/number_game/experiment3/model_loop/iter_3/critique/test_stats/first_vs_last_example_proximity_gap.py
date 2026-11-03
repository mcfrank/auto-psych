# name: first_vs_last_example_proximity_gap
# description: On 2+ example sets (target not an example), yes rate when the target is nearer the first-shown example minus when nearer the last-shown example; observed above null means the first example anchors generalisation more than the order-blind model predicts (primacy), below null means recency.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).values
    n = ex.notna().sum(1).values
    first = ex.iloc[:, 0].values
    last = ex.values[np.arange(len(ex)), n - 1]
    isex = (ex.values == t[:, None]).any(1)
    d1 = np.abs(t - first); dl = np.abs(t - last)
    r = df["rating"].astype(float).values
    m = (n >= 2) & ~isex
    a = r[m & (d1 < dl)]; b = r[m & (dl < d1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
