# name: within_set_magnitude_slope
# description: OLS slope (per 100 units) of yes response on target value after demeaning both rating and target within each number_set, non-example targets only; observed < null means people favour smaller targets (a small-number bias) more than the model, observed > null means they favour larger targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    m = ~inset
    g = df["number_set"][m]
    tt = t[m] / 100.0
    rr = df["rating"].astype(float)[m]
    tc = tt - tt.groupby(g).transform("mean")
    rc = rr - rr.groupby(g).transform("mean")
    den = float((tc * tc).sum())
    if den <= 0:
        return 0.0
    return float((tc * rc).sum() / den)
