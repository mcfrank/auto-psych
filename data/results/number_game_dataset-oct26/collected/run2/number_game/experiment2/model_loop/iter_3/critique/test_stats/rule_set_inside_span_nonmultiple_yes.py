# name: rule_set_inside_span_nonmultiple_yes
# description: In 2+-example sets whose gcd is >= 3, yes-rate for targets strictly inside the example span that are NOT multiples of the gcd; observed below the null means the model's filled-in span over-endorses rule-violating in-range numbers (people let the multiples rule override size), above means people accept in-range non-multiples even more than modelled.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    info = {}
    for s in sets:
        e = [int(v) for v in s.split(",")]
        g = 0
        for v in e:
            g = math.gcd(g, v)
        info[s] = (len(e), g, min(e), max(e))
    k = ns.map({s: v[0] for s, v in info.items()}).values
    g = ns.map({s: v[1] for s, v in info.items()}).values
    lo = ns.map({s: v[2] for s, v in info.items()}).values
    hi = ns.map({s: v[3] for s, v in info.items()}).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    gs = np.where(g >= 3, g, 1)
    sel = (k >= 2) & (g >= 3) & (t > lo) & (t < hi) & (t % gs != 0)
    if sel.sum() == 0:
        return 0.0
    return float(y[sel].mean())
