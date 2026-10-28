# name: trial_drift_slope
# description: OLS slope (per 100 trials) of rating minus its (number_set, target)-cell mean on trial_index; nonzero observed vs null_mean means a drift in willingness to say yes over the session that the stationary model lacks (observed < null_mean: people become more conservative over time).
def test_statistic(df):
    r = df["rating"].astype(float)
    res = (r - r.groupby([df["number_set"], df["target"]]).transform("mean")).to_numpy()
    x = df["trial_index"].to_numpy().astype(float) / 100.0
    x = x - x.mean()
    den = (x * x).sum()
    if den == 0:
        return 0.0
    return float((x * res).sum() / den)
