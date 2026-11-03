# name: near_target_distance_gradient
# description: Yes-rate for targets within 5 of the nearest example minus yes-rate for targets 6-20 away; observed > null means people show a stronger numeric-similarity gradient than the model predicts, observed < null a weaker one.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy()[:, None]
    dmin = np.nanmin(np.abs(ex - t), axis=1)
    r = df["rating"].to_numpy(float)
    a = (dmin >= 1) & (dmin <= 5)
    b = (dmin >= 6) & (dmin <= 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
