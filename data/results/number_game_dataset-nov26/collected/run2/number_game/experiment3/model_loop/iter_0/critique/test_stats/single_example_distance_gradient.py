# name: single_example_distance_gradient
# description: For one-example sets, yes rate for targets within 10 of the example minus yes rate for targets more than 30 away; observed > null means people's magnitude generalization from a lone example is steeper than the model's, observed < null shallower.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    one = n == 1
    e = pd.to_numeric(df["number_set"].astype(str).where(one, "0"), errors="coerce").fillna(0)
    d = (df["target"].astype(int) - e).abs()
    r = df["rating"].astype(float)
    a = r[one & (d <= 10)]; b = r[one & (d > 30)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
