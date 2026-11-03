# name: single_example_distance_gradient
# description: For one-example sets, yes rate on targets within 10 of the example minus yes rate on targets 40 or more away; observed above null means the model under-produces magnitude-similarity generalisation from a single example (too flat), below null means it over-produces it.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",").to_numpy()
    x = pd.to_numeric(s.where(one, "0")).to_numpy()
    t = df["target"].astype(int).to_numpy()
    d = np.abs(t - x)
    r = df["rating"].to_numpy().astype(float)
    near = one & (d <= 10) & (d > 0)
    far = one & (d >= 40)
    if not near.any() or not far.any():
        return 0.0
    return float(r[near].mean() - r[far].mean())
