# name: singleton_near_far_gradient
# description: For single-example sets, yes rate to targets within 10 of the example minus yes rate to targets more than 10 away; observed above null means people generalise more locally (steeper similarity gradient) than the model predicts, below means flatter.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    r = df["rating"].values
    d = np.abs(t - x)
    near = single.values & (d <= 10) & (d > 0)
    far = single.values & (d > 10)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
