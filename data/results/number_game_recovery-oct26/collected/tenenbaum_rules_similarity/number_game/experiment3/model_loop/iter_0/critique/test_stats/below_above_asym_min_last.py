# name: below_above_asym_min_last
# description: In sets of >=2 whose smallest example is shown last, yes-rate for targets 1-10 below the min minus yes-rate for targets 1-10 above the max; observed below null means a last-shown minimum also marks a sharp lower edge, which the model (edge credit only for a last max) misses.
def test_statistic(df):
    sp = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",")
    n = sp.str.len().values
    last = sp.str[-1].astype(int).values
    mx = sp.apply(lambda v: max(int(x) for x in v)).values
    mn = sp.apply(lambda v: min(int(x) for x in v)).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    g = (n >= 2) & (last == mn)
    dn = g & (t < mn) & (t >= mn - 10)
    up = g & (t > mx) & (t <= mx + 10)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[dn].mean() - r[up].mean())
