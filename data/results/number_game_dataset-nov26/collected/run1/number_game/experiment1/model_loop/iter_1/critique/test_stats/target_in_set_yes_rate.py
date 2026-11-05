# name: target_in_set_yes_rate
# description: Yes rate when the target is one of the shown examples; observed above null means the model under-predicts near-certain endorsement of examples.
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
    return m(dist == 0)
