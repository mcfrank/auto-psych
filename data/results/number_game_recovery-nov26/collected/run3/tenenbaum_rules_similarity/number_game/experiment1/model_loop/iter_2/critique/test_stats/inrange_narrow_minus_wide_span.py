# name: inrange_narrow_minus_wide_span
# description: Yes-rate on non-example targets lying inside [min,max] of the examples, for sets whose span is <=30 minus sets whose span is >=60; observed above model means people fill in narrow ranges more (relative to wide ones) than the Erlang interval-size prior predicts, below means wide ranges are filled in relatively more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    parsed = {s: [int(x) for x in s.split(",")] for s in sets.unique()}
    lo = sets.map({s: min(v) for s, v in parsed.items()}).to_numpy()
    hi = sets.map({s: max(v) for s, v in parsed.items()}).to_numpy()
    n = sets.map({s: len(v) for s, v in parsed.items()}).to_numpy()
    t = df["target"].astype(int).to_numpy()
    inset = np.array([x in parsed[s] for s, x in zip(sets.to_numpy(), t)])
    r = df["rating"].to_numpy(dtype=float)
    inside = (t > lo) & (t < hi) & ~inset & (n >= 2)
    span = hi - lo
    a = r[inside & (span <= 30)]
    b = r[inside & (span >= 60)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
