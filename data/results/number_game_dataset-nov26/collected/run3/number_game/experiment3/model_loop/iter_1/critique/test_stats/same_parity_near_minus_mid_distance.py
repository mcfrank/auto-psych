# name: same_parity_near_minus_mid_distance
# description: Among non-example targets at an even distance (same parity) from their nearest example, yes rate at distance 2-4 minus yes rate at distance 6-30; observed below null means the model over-produces a proximity gradient (people do not favour near same-parity numbers over moderately distant ones).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ev = (dist % 2 == 0) & ~is_ex
    a = ev & (dist >= 2) & (dist <= 4)
    b = ev & (dist >= 6) & (dist <= 30)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
