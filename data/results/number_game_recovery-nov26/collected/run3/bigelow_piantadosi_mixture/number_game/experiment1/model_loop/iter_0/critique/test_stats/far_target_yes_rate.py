# name: far_target_yes_rate
# description: Yes rate among targets more than 20 away from every example and sharing no last digit with any example; observed > null means the model under-produces diffuse/random "yes" to unrelated targets (missing lapse/broad generalization).
def test_statistic(df):
    k = df["number_set"].str.count(",") + 1
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    d = np.abs(ex - t[:, None])
    mind = np.nanmin(d, axis=1)
    lastmatch = np.nansum((np.mod(ex, 10) == np.mod(t, 10)[:, None]) & ~np.isnan(ex), axis=1) > 0
    m = (mind > 20) & ~lastmatch
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
