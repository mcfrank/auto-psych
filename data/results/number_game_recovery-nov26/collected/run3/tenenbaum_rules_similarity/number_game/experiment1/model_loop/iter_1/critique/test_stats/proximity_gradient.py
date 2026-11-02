# name: proximity_gradient
# description: Yes-rate for targets within 5 of the nearest example minus yes-rate for targets more than 20 from the nearest example; observed > model means people generalise by numeric similarity/magnitude more steeply than the rule+interval posterior predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    tgt = df["target"].astype(float).to_numpy()
    dist = np.nanmin(np.abs(sets - tgt[:, None]), axis=1)
    r = df["rating"].to_numpy().astype(float)
    near, far = dist <= 5, dist > 20
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
