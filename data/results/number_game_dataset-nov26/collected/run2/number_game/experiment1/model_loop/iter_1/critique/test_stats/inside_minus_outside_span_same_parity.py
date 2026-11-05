# name: inside_minus_outside_span_same_parity
# description: For all-even or all-odd sets, yes-rate on same-parity targets inside [min,max] of the examples minus yes-rate on same-parity targets outside it; observed > null means people show a stronger range restriction than the model, observed < null a weaker one.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    u = sets.unique()
    info = {}
    for s in u:
        xs = [int(x) for x in s.split(",")]
        par = set(x % 2 for x in xs)
        info[s] = (min(xs), max(xs), par.pop() if len(par) == 1 else -1)
    lo = sets.map({k: v[0] for k, v in info.items()}).values
    hi = sets.map({k: v[1] for k, v in info.items()}).values
    pa = sets.map({k: v[2] for k, v in info.items()}).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    same = (pa >= 0) & (t % 2 == pa)
    ins = same & (t >= lo) & (t <= hi)
    out = same & ((t < lo) | (t > hi))
    if ins.sum() == 0 or out.sum() == 0:
        return 0.0
    return float(r[ins].mean() - r[out].mean())
