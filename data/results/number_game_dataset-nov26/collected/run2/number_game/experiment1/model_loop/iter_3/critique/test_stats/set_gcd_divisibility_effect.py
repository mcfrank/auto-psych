# name: set_gcd_divisibility_effect
# description: In multi-example sets whose examples' greatest common divisor g is >= 3, yes-rate for targets divisible by g minus targets not divisible by g; observed > null means people generalize along the examples' exact common divisor (e.g. 15/45 for 90,45; 28 for 84,56; 13 for 13,91), which the model's fixed multiple-of-3..12 list misses (observed < null: the reverse).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    g = ex.map(lambda l: math.gcd(*[int(x) for x in l]) if len(l) > 1 else int(l[0])).values.astype(int)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ok = (n >= 2) & (g >= 3)
    div = (t % np.maximum(g, 1)) == 0
    a = ok & div
    b = ok & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
