# name: round_factor_tens_vs_fives
# description: Among sets whose examples' gcd is exactly 5 (e.g. 5; 5,65; 90,100,5), yes rate on targets that are multiples of 10 minus yes rate on odd multiples of 5 (both are multiples of the common factor, which the model credits equally); observed > null means people favour round tens over fives beyond what the model predicts.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    g = {s: reduce(math.gcd, [int(x) for x in s.split(",")]) for s in sets}
    gg = df["number_set"].map(g).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    m = gg == 5
    a = m & (t % 10 == 0)
    b = m & (t % 5 == 0) & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
