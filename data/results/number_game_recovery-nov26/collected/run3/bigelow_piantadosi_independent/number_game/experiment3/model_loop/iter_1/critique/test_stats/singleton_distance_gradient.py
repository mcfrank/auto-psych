# name: singleton_distance_gradient
# description: Among single-example sets, yes rate for targets within 10 of the example (excluding the example itself) minus yes rate for targets more than 30 away; observed below null means people generalise less locally (flatter similarity gradient) than the model's interval mechanism predicts, above null means a steeper gradient.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    if single.sum() == 0:
        return 0.0
    x = pd.to_numeric(s[single], errors="coerce").to_numpy()
    t = df.loc[single, "target"].astype(int).to_numpy()
    r = df.loc[single, "rating"].astype(float).to_numpy()
    dist = np.abs(t - x)
    near = (dist > 0) & (dist <= 10)
    far = dist > 30
    a = r[near].mean() if near.any() else 0.0
    b = r[far].mean() if far.any() else 0.0
    return float(a - b)
