# name: recency_last_vs_first_example
# description: For sets of 2+ examples and non-example targets within 10 of the first- or last-shown example, yes-rate on targets strictly closer to the last-shown example minus yes-rate on targets strictly closer to the first-shown; observed > model means people generalize more around the most recent example (recency) than the order-blind model predicts, observed < model means a primacy bias.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), np.maximum(n - 1, 0)]
    df_ = np.abs(t - first); dl = np.abs(t - last)
    is_ex = np.any(ex == t[:, None], axis=1)
    base = (n >= 2) & ~is_ex & (np.minimum(df_, dl) <= 10)
    a = base & (dl < df_); b = base & (df_ < dl)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
