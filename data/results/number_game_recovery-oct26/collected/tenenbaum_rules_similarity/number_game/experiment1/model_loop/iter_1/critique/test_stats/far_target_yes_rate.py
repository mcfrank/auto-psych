# name: far_target_yes_rate
# description: Yes rate for targets at least 30 away from every shown example; the model has no lapse/guessing term, so observed above null_mean means people say yes to distant targets more than the model allows.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    d = np.nanmin(np.abs(s - t[:, None]), axis=1)
    m = d >= 30
    return float(r[m].mean()) if m.any() else 0.0
