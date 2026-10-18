# name: far_target_yes_rate
# description: Yes rate for targets more than 25 away from every example (magnitude-distant targets); observed above null means the model under-predicts acceptance of numerically distant targets (people generalize by rule, not by magnitude proximity), below means it over-predicts.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[:, None]
    d = np.nanmin(np.abs(s - t), axis=1)
    r = df["rating"].astype(float).to_numpy()
    m = d > 25
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
