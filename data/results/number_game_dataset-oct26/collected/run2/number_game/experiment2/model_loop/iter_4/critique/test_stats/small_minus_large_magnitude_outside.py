# name: small_minus_large_magnitude_outside
# description: Yes-rate for outside-span targets 1-15 beyond the span in sets whose examples are all <= 30, minus the same in sets whose examples are all >= 60; observed below the null means people generalize less (in absolute units) around small numbers than large ones — log/Weber-scaled magnitude similarity — which the model's linear size mismatch under-produces (above means the opposite).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: [int(v) for v in s.split(",")] for s in sets}
    hi = ns.map({s: max(e) for s, e in ex.items()}).values
    lo = ns.map({s: min(e) for s, e in ex.items()}).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    d = np.maximum(np.maximum(lo - t, t - hi), 0)
    near = (d >= 1) & (d <= 15)
    small = near & (hi <= 30)
    large = near & (lo >= 60)
    if small.sum() == 0 or large.sum() == 0:
        return 0.0
    return float(y[small].mean() - y[large].mean())
