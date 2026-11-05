# name: outside_range_distance_corr
# description: Among multi-example sets, for targets outside [min,max] of the examples, Pearson correlation of yes with distance to the nearest range bound; observed more negative than null means endorsement decays with distance beyond the range more gradually/strongly than the model's step-like bounds produce.
def test_statistic(df):
    sets = df["number_set"].unique()
    mn = {s: min(int(x) for x in str(s).split(",")) for s in sets}
    mx = {s: max(int(x) for x in str(s).split(",")) for s in sets}
    k = df["number_set"].astype(str).str.count(",") + 1
    lo = df["number_set"].map(mn).values
    hi = df["number_set"].map(mx).values
    t = df["target"].astype(int).values
    dist = np.maximum(lo - t, 0) + np.maximum(t - hi, 0)
    sel = (k.values >= 2) & (dist > 0)
    if sel.sum() < 3:
        return 0.0
    d = dist[sel].astype(float)
    r = df["rating"].values[sel].astype(float)
    if d.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(d, r)[0, 1])
