# name: small_number_generalization_advantage
# description: Among non-member targets within 1-10 of the nearest example, yes rate when that example is <=30 minus when it is >=70; observed > null means people generalise more widely around small numbers (log-like number line) than the model's linear-width ranges predict, observed < null the reverse.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (d >= 1) & (d <= 10)
    a, b = m & (ne <= 30), m & (ne >= 70)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
