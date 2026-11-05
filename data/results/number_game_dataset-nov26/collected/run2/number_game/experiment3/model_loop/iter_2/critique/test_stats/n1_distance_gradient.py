# name: n1_distance_gradient
# description: For one-example sets, yes rate for targets within 10 of the example minus targets more than 30 away; observed > null means people's magnitude generalization from a lone example is steeper than the model's, observed < null shallower (model over-produces nearness).
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    e = pd.to_numeric(s.where(one, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    d = np.abs(t - e)
    near = one & (d <= 10) & (d > 0)
    far = one & (d > 30)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
