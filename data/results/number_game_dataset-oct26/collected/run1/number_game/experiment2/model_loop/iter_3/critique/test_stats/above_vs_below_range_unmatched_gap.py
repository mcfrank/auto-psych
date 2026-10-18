# name: above_vs_below_range_unmatched_gap
# description: In multi-example sets, among targets matching no example's final digit and outside [min,max] of the examples, yes rate for targets above the max minus targets below the min; t_observed above null_mean means people extrapolate upward more than downward (a magnitude asymmetry the symmetric decade-band term cannot produce); below means downward extrapolation is favoured.
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        xs = [int(x) for x in s.split(",")]
        info[s] = (len(xs) >= 2, min(xs), max(xs), set(x % 10 for x in xs))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    multi = np.array([info[s][0] for s in sets])
    lo = np.array([info[s][1] for s in sets])
    hi = np.array([info[s][2] for s in sets])
    unm = np.array([(v % 10) not in info[s][3] for s, v in zip(sets, t)])
    a = multi & unm & (t > hi)
    b = multi & unm & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
