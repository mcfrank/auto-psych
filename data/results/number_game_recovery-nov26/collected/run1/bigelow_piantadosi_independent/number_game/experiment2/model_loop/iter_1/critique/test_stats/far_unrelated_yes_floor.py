# name: far_unrelated_yes_floor
# description: Yes rate for targets more than 20 from every example, outside the example range, on sets whose examples do not share a parity; observed > model means a lapse/guessing floor of yes responses the concept posterior does not predict.
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
    p = np.nanmax(ex % 2, axis=1) == np.nanmin(ex % 2, axis=1)
    m = (dist > 20) & ((t < mn) | (t > mx)) & (~p)
    return rate(m)
