# name: n1_distance_gradient
# description: For one-example sets, yes rate for targets within 10 of the example minus yes rate for targets more than 30 away; observed > null means the model still makes lone-example generalization too flat in magnitude, observed < null means the model's band is still steeper than people's (who are nearly flat in distance).
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    e = pd.to_numeric(s.where(one, "0"), errors="coerce").fillna(0).values
    d = np.abs(df["target"].astype(int).values - e)
    r = df["rating"].astype(float).values
    a = one & (d <= 10)
    b = one & (d > 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
