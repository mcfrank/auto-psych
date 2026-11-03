# name: example_target_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed < model means people reject examples more often than the model's near-certain membership implies (a response-level lapse).
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
    return rate(dist == 0)
