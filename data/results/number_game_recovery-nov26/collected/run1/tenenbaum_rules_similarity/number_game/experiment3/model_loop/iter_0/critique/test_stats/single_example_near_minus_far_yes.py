# name: single_example_near_minus_far_yes
# description: On single-example sets, yes rate for targets within 10 of the example minus yes rate for targets more than 30 away; observed above null means people's one-shot generalisation is more similarity/proximity-graded than the model predicts (model too rule-driven), below means the model overweights proximity.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single), errors="coerce").to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    d = np.abs(t - x)
    near = single.to_numpy() & (d > 0) & (d <= 10)
    far = single.to_numpy() & (d > 30)
    if not near.any() or not far.any():
        return 0.0
    return float(y[near].mean() - y[far].mean())
