# name: magnitude_scaled_neighbour_breadth
# description: In multi-example sets, among non-example targets 4-15 from their nearest example and sharing no final digit with any example, yes-rate when the nearest example is >= 50 minus when it is <= 30; observed above the null means people's numeric neighbourhoods widen with magnitude (Weber-like) more than the model's compressed number line predicts, below means the model over-widens them at large magnitudes.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    diff = np.abs(E - t[:, None])
    dist = np.nanmin(diff, axis=1)
    nearest = E[np.arange(len(t)), np.nanargmin(np.where(np.isnan(diff), np.inf, diff), axis=1)]
    share_fd = np.any(np.mod(E, 10) == np.mod(t, 10)[:, None], axis=1)
    m = (n >= 2) & (dist >= 4) & (dist <= 15) & (~share_fd)
    a, b = m & (nearest >= 50), m & (nearest <= 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
