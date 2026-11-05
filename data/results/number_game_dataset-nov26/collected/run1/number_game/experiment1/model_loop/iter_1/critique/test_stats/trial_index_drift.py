# name: trial_index_drift
# description: Correlation between trial_index and yes response; observed differing from null means response bias drifts over the session in a way the model does not capture.
def test_statistic(df):
    S = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(S), axis=1)
    Si = np.nan_to_num(S, nan=0.0).astype(int)
    dist = np.nanmin(np.abs(S - t[:, None]), axis=1)
    smin = np.nanmin(S, axis=1); smax = np.nanmax(S, axis=1)
    def m(mask):
        return float(np.mean(r[mask])) if np.sum(mask) > 0 else 0.0
    ti = df["trial_index"].astype(float).to_numpy()
    if np.std(ti) == 0 or np.std(r) == 0:
        return 0.0
    v = float(np.corrcoef(ti, r)[0, 1])
    return v if np.isfinite(v) else 0.0
