# name: upward_minus_downward_generalization
# description: For targets 1-15 outside the example range, yes rate above the maximum minus yes rate below the minimum; observed > model means people extend more upward (or less downward) than the model's compressed number line predicts.
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
    up = (t > mx) & (t - mx <= 15)
    dn = (t < mn) & (mn - t <= 15)
    return rate(up) - rate(dn)
