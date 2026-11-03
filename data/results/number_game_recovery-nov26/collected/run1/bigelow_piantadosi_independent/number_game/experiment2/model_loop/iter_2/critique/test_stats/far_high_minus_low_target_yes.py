# name: far_high_minus_low_target_yes
# description: Among targets more than 10 from every example, yes rate for targets >= 70 minus targets <= 30; observed > null means people favour large numbers (or the model over-favours small ones via its compressed number line), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    far = dist > 10
    return rate(far & (t >= 70)) - rate(far & (t <= 30))
