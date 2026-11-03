# name: within_set_target_magnitude_slope
# description: Least-squares slope of the set-demeaned yes response on set-demeaned target/100 (pooled over all trials); observed below null means people favour small targets within a set more than the model predicts (a magnitude/small-number bias), above null means they favour large targets more.
def test_statistic(df):
    t = df["target"].astype(float) / 100.0
    r = df["rating"].astype(float)
    key = df["number_set"].astype(str)
    tc = t - t.groupby(key).transform("mean")
    rc = r - r.groupby(key).transform("mean")
    den = float((tc * tc).sum())
    if den <= 0:
        return 0.0
    return float((tc * rc).sum() / den)
