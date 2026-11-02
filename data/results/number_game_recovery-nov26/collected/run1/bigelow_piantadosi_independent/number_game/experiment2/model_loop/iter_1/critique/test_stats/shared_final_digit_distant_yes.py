# name: shared_final_digit_distant_yes
# description: Among sets whose examples all end in the same digit, yes rate for non-example targets with that final digit lying more than 10 from every example; observed > model means people commit to the 'ends in d' rule more firmly than the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    ld = ex % 10
    a = np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)
    m = a & ((ex[:, 0] % 10) == (t % 10)) & (dist > 10)
    return rate(m)
