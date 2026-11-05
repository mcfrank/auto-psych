# name: multi_outside_span_far_near_gap
# description: On 2-4-example sets, yes rate for targets far outside the examples' range (more than 30 beyond the nearest end) minus targets just outside (1-10 beyond); observed > null means people's endorsement falls off with distance outside the range less steeply than the model's Gaussian band (observed < null: falls off more steeply).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    lo = np.array([min(int(x) for x in s.split(",")) for s in u])
    hi = np.array([max(int(x) for x in s.split(",")) for s in u])
    nn = np.array([s.count(",") + 1 for s in u])
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    gap = np.maximum(0, np.maximum(lo[inv] - t, t - hi[inv]))
    m = nn[inv] >= 2
    far = m & (gap > 30)
    near = m & (gap >= 1) & (gap <= 10)
    if far.sum() == 0 or near.sum() == 0:
        return 0.0
    return float(r[far].mean() - r[near].mean())
