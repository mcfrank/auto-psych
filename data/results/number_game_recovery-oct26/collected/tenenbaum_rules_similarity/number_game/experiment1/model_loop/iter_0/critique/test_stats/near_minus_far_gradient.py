# name: near_minus_far_gradient
# description: Yes-rate for non-example targets within 3 of the nearest example minus yes-rate for targets more than 40 away; observed below the null means the model's generalisation gradient over numerical distance is too steep (humans are flatter), above means too shallow.
def test_statistic(df):
    arr = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values[:, None]
    dist = np.nanmin(np.abs(arr - t), axis=1)
    r = df["rating"].values.astype(float)
    near = (dist > 0) & (dist <= 3)
    far = dist > 40
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
