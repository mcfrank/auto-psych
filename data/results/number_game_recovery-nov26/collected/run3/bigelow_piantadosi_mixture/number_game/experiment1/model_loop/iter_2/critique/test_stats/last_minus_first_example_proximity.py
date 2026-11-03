# name: last_minus_first_example_proximity
# description: Yes-rate for non-example targets within 5 of the LAST shown example minus within 5 of the FIRST shown example (targets near both excluded, multi-example sets only); the model is order-blind so its null is ~0, a positive observed value means a recency effect, negative a primacy effect the model lacks.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int).values
    first = ex.iloc[:, 0].values
    last = ex.values[np.arange(len(ex)), n.values - 1]
    is_ex = ex.eq(df["target"].astype(int), axis=0).any(axis=1).values
    df_ = np.abs(t - first)
    dl = np.abs(t - last)
    base = (n.values >= 2) & (~is_ex)
    near_last = base & (dl <= 5) & (df_ > 5)
    near_first = base & (df_ <= 5) & (dl > 5)
    r = df["rating"].astype(float).values
    if near_last.sum() == 0 or near_first.sum() == 0:
        return 0.0
    return float(r[near_last].mean() - r[near_first].mean())
