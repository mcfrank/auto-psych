# name: single_example_near_generalization
# description: Yes rate for targets within 10 of the nearest example on 1-example sets minus the same on 4-example sets; observed > null means people loosen generalization with few examples more than the model's size principle implies (observed < null: less).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].astype(float).to_numpy()
    a = (n == 1) & (d >= 1) & (d <= 10)
    b = (n == 4) & (d >= 1) & (d <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
