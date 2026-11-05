# name: in_minus_out_range_3plus
# description: For sets with 3+ examples, non-example-target yes rate inside [min,max] of the examples minus outside it; observed above null means people confine generalization to the examples' range more than the model (below = model over-confines).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    base = (n >= 3) & (~isex)
    inn = base & (t >= mn) & (t <= mx); out = base & ((t < mn) | (t > mx))
    if not inn.any() or not out.any(): return 0.0
    return float(r[inn].mean() - r[out].mean())
