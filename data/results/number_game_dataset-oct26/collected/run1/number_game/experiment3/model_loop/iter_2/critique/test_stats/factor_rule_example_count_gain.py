# name: factor_rule_example_count_gain
# description: Among multi-example sets with common factor gcd>=3, yes rate on targets that are multiples of the gcd for sets of 3-4 examples minus for sets of 2 examples; observed > null means more examples strengthen the factor rule (size principle) more than the model's constant factor boost predicts.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    g = {s: reduce(math.gcd, [int(x) for x in s.split(",")]) for s in sets}
    gg = df["number_set"].map(g).values
    n = (df["number_set"].str.count(",") + 1).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    mult = (gg >= 3) & (t % np.maximum(gg, 1) == 0)
    a = mult & (n >= 3)
    b = mult & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
