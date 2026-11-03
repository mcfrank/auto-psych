# name: trial_index_drift
# description: Pearson correlation between trial_index and rating; observed more negative than null means humans say yes less often as the session goes on, a drift the model lacks.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    ti = df["trial_index"].astype(float).to_numpy()
    if np.std(ti) == 0 or np.std(r) == 0:
        return 0.0
    return float(np.corrcoef(ti, r)[0, 1])
