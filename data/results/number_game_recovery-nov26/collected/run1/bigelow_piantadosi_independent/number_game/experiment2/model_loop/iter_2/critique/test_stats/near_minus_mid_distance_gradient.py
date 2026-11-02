# name: near_minus_mid_distance_gradient
# description: For sets of >= 2 examples, yes rate for non-example targets 1-2 from the nearest example minus those 6-15 away; observed > null means people's similarity gradient falls off more steeply than the model's interval mixture, observed < null means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    return rate((n >= 2) & nonex & (dist <= 2)) - rate((n >= 2) & (dist >= 6) & (dist <= 15))
