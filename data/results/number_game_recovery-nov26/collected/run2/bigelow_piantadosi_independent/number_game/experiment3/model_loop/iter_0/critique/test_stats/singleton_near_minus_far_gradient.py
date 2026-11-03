# name: singleton_near_minus_far_gradient
# description: For one-example sets and targets other than the example, yes-rate when |target - example| <= 10 minus when it is > 10; observed above null_mean means people show a steeper magnitude-similarity gradient around a single example than the model's interval hypotheses produce (below means shallower).
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "nan"), errors="coerce").to_numpy()
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    d = np.abs(t - x)
    ok = single.to_numpy() & (d > 0)
    near = ok & (d <= 10)
    far = ok & (d > 10)
    if not near.any() or not far.any():
        return 0.0
    return float(r[near].mean() - r[far].mean())
