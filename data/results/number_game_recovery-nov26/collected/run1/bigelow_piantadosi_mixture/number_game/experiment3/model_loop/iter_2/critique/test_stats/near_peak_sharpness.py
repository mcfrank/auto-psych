# name: near_peak_sharpness
# description: Among sets of >=2 examples, yes rate for non-member targets 1-2 from the nearest example minus yes rate for targets 6-10 away; observed > null means generalisation falls off more sharply right around the examples than the model's gamma width prior predicts, observed < null that it is flatter.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    a = (n >= 2) & (d >= 1) & (d <= 2)
    b = (n >= 2) & (d >= 6) & (d <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
