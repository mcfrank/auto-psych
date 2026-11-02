# name: trial_index_yes_slope
# description: OLS slope of the yes response on standardized trial_index across all trials; observed below null means yes rate declines over the session more than the (stationary) model predicts, above means it rises.
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    sd = x.std()
    if sd == 0:
        return 0.0
    z = (x - x.mean()) / sd
    return float(np.mean(z * (y - y.mean())))
