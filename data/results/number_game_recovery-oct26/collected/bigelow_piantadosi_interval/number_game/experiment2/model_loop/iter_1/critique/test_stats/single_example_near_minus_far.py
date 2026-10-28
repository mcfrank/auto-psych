# name: single_example_near_minus_far
# description: For single-example sets, yes-rate for targets within 10 of the example (excluding the example) minus yes-rate for targets more than 30 away; observed above the null means people's generalization from one number is more sharply local than the model predicts, below means it is flatter (broader) than the model predicts.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    e = pd.to_numeric(ns.where(single, "nan"), errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    dist = np.abs(t - e)
    r = df["rating"].astype(float).to_numpy()
    s = single.to_numpy()
    near = s & (dist > 0) & (dist <= 10)
    far = s & (dist > 30)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
