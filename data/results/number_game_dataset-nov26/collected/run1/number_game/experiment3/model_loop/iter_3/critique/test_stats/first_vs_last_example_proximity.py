# name: first_vs_last_example_proximity
# description: In 2-4-example sets, yes-rate for targets within 10 of the first-shown example (and >10 from the last) minus yes-rate for targets within 10 of the last-shown example (and >10 from the first); observed > null means people weight the first example more than the order-blind model (primacy), observed < null means a recency weighting.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    v = s.values
    t = df["target"].astype(int).values
    n = (~np.isnan(v)).sum(axis=1)
    first = v[:, 0]
    last = v[np.arange(len(v)), np.maximum(n - 1, 0)]
    df_ = np.abs(t - first)
    dl = np.abs(t - last)
    r = df["rating"].values
    a = (n >= 2) & (df_ <= 10) & (dl > 10)
    b = (n >= 2) & (dl <= 10) & (df_ > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
