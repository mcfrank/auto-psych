# name: first_vs_last_example_proximity
# description: In 2-4 example sets, yes-rate for non-example targets within 10 of the first-shown example (and farther from the last) minus yes-rate for targets within 10 of the last-shown example (and farther from the first); nonzero observed vs a model null means presentation order matters (positive = primacy), which the order-blind model cannot produce.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    last = E[np.arange(len(E)), np.maximum(n - 1, 0)]
    first = E[:, 0]
    is_ex = np.any(E == t[:, None], axis=1)
    df1 = np.abs(t - first); dl = np.abs(t - last)
    base = (n >= 2) & ~is_ex
    a = base & (df1 <= 10) & (dl > 10)
    b = base & (dl <= 10) & (df1 > 10)
    r = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
