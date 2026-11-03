# name: span_scaled_outside_generalization
# description: For sets of >=2 examples with no shared parity, yes rate for targets just outside the span (distance <= 25% of span, at least 1) in wide sets (span>=40) minus the same in narrow sets (span<=15); above the null means generalization beyond the examples scales with their spread more than the model's interval-size principle predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    arr = ex.apply(lambda v: [int(x) for x in v])
    lo = arr.map(min).values
    hi = arr.map(max).values
    n = arr.map(len).values
    npar = arr.map(lambda v: len({x % 2 for x in v})).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    span = hi - lo
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    lim = np.maximum(1, 0.25 * span)
    base = (n >= 2) & (npar == 2) & (dist > 0) & (dist <= lim)
    w = base & (span >= 40)
    nr = base & (span <= 15)
    if w.sum() == 0 or nr.sum() == 0:
        return 0.0
    return float(y[w].mean() - y[nr].mean())
