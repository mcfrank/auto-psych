# name: first_minus_last_example_endorsement
# description: On 3-4 example sets, yes rate when the target is the first-shown example minus when it is the last-shown example; observed above null means people favour earlier-shown examples (a primacy effect the order-invariant model cannot produce).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    idx = np.arange(len(t))
    last = E[idx, np.clip(n - 1, 0, 3)]
    big = n >= 3
    a = big & (E[:, 0] == t)
    b = big & (last == t)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
