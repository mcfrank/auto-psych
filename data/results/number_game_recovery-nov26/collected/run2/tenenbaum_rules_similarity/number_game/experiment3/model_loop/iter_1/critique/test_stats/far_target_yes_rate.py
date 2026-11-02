# name: far_target_yes_rate
# description: Yes-rate for targets more than 40 from every example; observed above null means the model under-produces broad, distance-insensitive generalisation (below means it over-produces it).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    dist = np.where(~np.isnan(ex), np.abs(ex - t[:, None]), np.inf).min(axis=1)
    m = dist > 40
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
