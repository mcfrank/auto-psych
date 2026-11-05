# name: between_set_yes_rate_spread
# description: Standard deviation across distinct number_sets of the per-set yes rate; observed above null means sets differ more in overall endorsement (rule vs similarity sets) than the model predicts.
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
    s = pd.Series(r).groupby(df["number_set"].astype(str).to_numpy()).mean()
    v = float(s.std(ddof=0)) if len(s) > 1 else 0.0
    return v if np.isfinite(v) else 0.0
