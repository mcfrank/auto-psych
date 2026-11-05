# name: unstructured_set_yes_rate
# description: Yes rate on 2+-example sets with no visible structure (mixed parity, no shared final digit, no common divisor >1); observed > null means people still generalize broadly from random-looking sets (model too stingy), observed < null they reject more than the model predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    u, inv = np.unique(sets.values, return_inverse=True)
    flag = np.zeros(len(u), dtype=bool)
    for i, s in enumerate(u):
        v = [int(x) for x in s.split(",")]
        if len(v) < 2:
            continue
        mixed = len({x % 2 for x in v}) > 1
        ld = len({x % 10 for x in v}) == 1
        g = 0
        for x in v:
            g = math.gcd(g, x)
        flag[i] = mixed and (not ld) and g <= 1
    m = flag[inv]
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values.astype(float)[m].mean())
