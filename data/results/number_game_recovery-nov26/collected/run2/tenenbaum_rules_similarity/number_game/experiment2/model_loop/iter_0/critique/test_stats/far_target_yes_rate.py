# name: far_target_yes_rate
# description: Yes rate for targets more than 30 away from every example; observed above the model's null means people say yes to remote targets more than the model's concepts (plus noise) allow, i.e. a baseline lapse/guess rate is missing.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(float).values[:, None]
    d = np.nanmin(np.abs(ex - t), axis=1)
    m = d > 30
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
