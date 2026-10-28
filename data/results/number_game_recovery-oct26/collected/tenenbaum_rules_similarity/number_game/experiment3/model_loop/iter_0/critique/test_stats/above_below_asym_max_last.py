# name: above_below_asym_max_last
# description: In sets of >=2 whose largest example is shown last, yes-rate for targets 1-10 above the max minus yes-rate for targets 1-10 below the min; observed below null means people extend upward even less (the model under-credits the top edge), above null means the model over-sharpens the top edge.
def test_statistic(df):
    sp = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",")
    n = sp.str.len().values
    last = sp.str[-1].astype(int).values
    mx = sp.apply(lambda v: max(int(x) for x in v)).values
    mn = sp.apply(lambda v: min(int(x) for x in v)).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    g = (n >= 2) & (last == mx)
    up = g & (t > mx) & (t <= mx + 10)
    dn = g & (t < mn) & (t >= mn - 10)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
