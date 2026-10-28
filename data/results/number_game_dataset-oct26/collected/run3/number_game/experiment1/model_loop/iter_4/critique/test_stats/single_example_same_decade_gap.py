# name: single_example_same_decade_gap
# description: For one-example sets, yes-rate on targets 1-9 away from the example that share its tens decade (target//10 == example//10, with 100 in decade 9) minus those 1-9 away in a different decade; observed above null means people group by decade more than the decade-bonus model predicts from one example, below means the model's decade bonus overshoots.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    e = pd.to_numeric(ns.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    d = np.abs(t - e)
    dec = lambda x: np.minimum(x, 99) // 10
    near = single.values & (d >= 1) & (d <= 9)
    same = near & (dec(t) == dec(e))
    diff = near & (dec(t) != dec(e))
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
