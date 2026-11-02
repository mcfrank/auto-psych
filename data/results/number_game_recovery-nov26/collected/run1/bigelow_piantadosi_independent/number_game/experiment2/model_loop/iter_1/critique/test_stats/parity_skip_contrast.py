# name: parity_skip_contrast
# description: Among sets whose examples all share a parity, yes rate for same-parity targets 2 away from the nearest example minus opposite-parity targets 1 away; observed > model means people use parity more sharply against neighbours than the model predicts.
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
    same = (ex[:, 0] % 2) == (t % 2)
    return rate(p & same & (dist == 2)) - rate(p & (~same) & (dist == 1))
