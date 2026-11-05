# name: near_minus_far_inside_range
# description: For sets with 3+ examples, among non-example targets inside [min,max], yes rate for targets within 3 of the nearest example minus those farther than 3; observed above null means people use numeric proximity more than the model (below = people disfavor near-neighbours relative to the model, i.e. less magnitude-based similarity).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    base = (n >= 3) & (~isex) & (t >= mn) & (t <= mx)
    a = base & (dist <= 3); b = base & (dist > 3)
    if not a.any() or not b.any(): return 0.0
    return float(r[a].mean() - r[b].mean())
