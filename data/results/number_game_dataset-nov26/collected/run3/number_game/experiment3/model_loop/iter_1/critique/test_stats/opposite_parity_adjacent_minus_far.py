# name: opposite_parity_adjacent_minus_far
# description: Among non-example targets at odd distance (opposite parity) from their nearest example, yes rate at distance 1 minus yes rate at distance 11+; observed below null means the model generalizes too much to immediate neighbours (n+-1) relative to far numbers.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    od = (dist % 2 == 1) & ~is_ex
    a = od & (dist == 1)
    b = od & (dist >= 11)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
