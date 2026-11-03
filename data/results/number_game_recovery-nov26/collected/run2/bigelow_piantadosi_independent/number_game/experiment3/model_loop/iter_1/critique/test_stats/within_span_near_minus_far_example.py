# name: within_span_near_minus_far_example
# description: Among non-example targets strictly inside the examples' span (sets of 2+ examples with span >= 15), yes-rate for targets within 3 of the nearest example minus yes-rate for targets 8+ from every example; observed above null_mean means people show a similarity gradient inside the span that the flat interval/rule model under-produces.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    lst = [np.array([int(x) for x in s.split(",")]) for s in df["number_set"].unique()]
    info = {s: a for s, a in zip(df["number_set"].unique(), lst)}
    ns = df["number_set"].to_numpy()
    lo = np.array([info[s].min() for s in ns]); hi = np.array([info[s].max() for s in ns])
    n = np.array([len(info[s]) for s in ns])
    ex = np.full((len(df), 4), -1000)
    for s, a in info.items():
        ex[ns == s, :len(a)] = a
    dmin = np.abs(ex - t[:, None]).min(axis=1)
    r = df["rating"].astype(float).to_numpy()
    base = (n >= 2) & (hi - lo >= 15) & (t > lo) & (t < hi) & (dmin > 0)
    near = base & (dmin <= 3); far = base & (dmin >= 8)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
