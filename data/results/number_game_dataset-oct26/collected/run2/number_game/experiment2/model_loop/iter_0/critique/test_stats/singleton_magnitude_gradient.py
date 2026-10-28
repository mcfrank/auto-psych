# name: singleton_magnitude_gradient
# description: For one-example sets (target not equal to the example), yes-rate for targets within 10 of the example minus yes-rate for targets more than 30 away; observed above the null means people's generalisation from a single example falls off with size distance more steeply than the model predicts, below means less steeply.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    x = pd.to_numeric(ns.where(single, "0"), errors="coerce").fillna(0).values
    t = df["target"].astype(int).values
    d = np.abs(t - x)
    r = df["rating"].astype(float).values
    s = single.values & (d > 0)
    near = s & (d <= 10)
    far = s & (d > 30)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
