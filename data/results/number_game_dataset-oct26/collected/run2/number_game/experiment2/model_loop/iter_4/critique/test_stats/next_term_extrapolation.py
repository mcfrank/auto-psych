# name: next_term_extrapolation
# description: In 2+-example sets with a nonzero last step, among outside-span targets 1-40 ahead of the span in the last step's direction, yes-rate for targets within 3 of the extrapolated next term (last + last step) minus the rest; observed above the null means people extrapolate the arithmetic step to predict the specific next number, which the model's smooth distance-based momentum cannot produce (below means no next-term peak, even less than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: [int(v) for v in s.split(",")] for s in sets}
    step = ns.map({s: (e[-1] - e[-2]) if len(e) >= 2 else 0 for s, e in ex.items()}).values
    la = ns.map({s: e[-1] for s, e in ex.items()}).values
    hi = ns.map({s: max(e) for s, e in ex.items()}).values
    lo = ns.map({s: min(e) for s, e in ex.items()}).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    ahead = ((step > 0) & (t > hi) & (t - hi <= 40)) | ((step < 0) & (t < lo) & (lo - t <= 40))
    nxt = la + step
    peak = ahead & (np.abs(t - nxt) <= 3)
    rest = ahead & ~peak
    if peak.sum() == 0 or rest.sum() == 0:
        return 0.0
    return float(y[peak].mean() - y[rest].mean())
