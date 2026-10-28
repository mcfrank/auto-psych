# name: in_set_target_yes_rate
# description: Yes rate when the target is one of the shown examples; observed below null_mean means people reject exact examples more often than the model (which predicts ~1) allows.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    m = np.any(s == t[:, None], axis=1)
    return float(r[m].mean()) if m.any() else 0.0
