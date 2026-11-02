# name: decade_boundary_effect
# description: Among non-member targets within 1-9 of the nearest example, yes rate when the target shares that example's decade (same (x-1)//10) minus when it is in a different decade; observed > null means people respect decade boundaries (tens-digit similarity) more than the model's continuous ranges predict, observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (d >= 1) & (d <= 9)
    same = (t - 1) // 10 == (ne - 1) // 10
    a, b = m & same, m & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
