# name: pair_out_of_span_yes_rate
# description: Yes-rate for targets outside the span of two-example sets (strictly below the smaller or above the larger example); observed below null_mean means the model over-generalizes beyond a pair of examples.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    m = (n == 2) & out
    r = df["rating"].astype(float).to_numpy()
    return float(r[m].mean()) if m.any() else 0.0
