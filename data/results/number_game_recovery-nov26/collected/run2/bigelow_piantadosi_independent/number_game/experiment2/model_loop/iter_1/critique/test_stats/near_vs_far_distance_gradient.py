# name: near_vs_far_distance_gradient
# description: Among targets not in the set, yes-rate for targets within 3 of the nearest example minus yes-rate for targets more than 10 from the nearest example; observed > null means people show a steeper numerical-similarity gradient than the model predicts.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].astype(float).to_numpy()
    near = (dist > 0) & (dist <= 3)
    far = dist > 10
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
