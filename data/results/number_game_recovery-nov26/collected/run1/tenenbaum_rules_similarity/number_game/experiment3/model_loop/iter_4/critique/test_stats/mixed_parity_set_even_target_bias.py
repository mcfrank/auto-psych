# name: mixed_parity_set_even_target_bias
# description: For sets containing both odd and even examples, yes rate for even non-example targets minus odd non-example targets; the model predicts near zero, so observed above null means people carry an even-number preference the model lacks, below null an odd preference.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        cache[x] = (set(v), len(set(a % 2 for a in v)) == 2)
    ex = np.array([ti in cache[x][0] for ti, x in zip(t, s)])
    mixed = np.array([cache[x][1] for x in s])
    even = (t % 2 == 0)
    a = y[mixed & ~ex & even]
    b = y[mixed & ~ex & ~even]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
