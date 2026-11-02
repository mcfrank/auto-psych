# name: example_target_endorsement_by_set_size
# description: Yes-rate on targets that are one of the shown examples in 3-4-example sets minus in 1-example sets; observed below null_mean means people disown examples of larger sets (treat some as strays) more than the model's constant 10% stray rate implies (above = less).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    isex = (ex == t[:, None]).any(axis=1)
    y = df["rating"].astype(float).to_numpy()
    a = isex & (n >= 3)
    b = isex & (n == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
