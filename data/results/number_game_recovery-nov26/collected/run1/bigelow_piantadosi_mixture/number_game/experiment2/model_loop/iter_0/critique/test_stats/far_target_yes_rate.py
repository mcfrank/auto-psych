# name: far_target_yes_rate
# description: Yes rate for targets more than 40 away from every example; observed above null_mean means people say yes to remote numbers more than the model's rules/intervals allow (missing lapse/guessing floor), below means the model over-generalises far away.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    m = dist > 40
    return float(r[m].mean()) if m.any() else 0.0
