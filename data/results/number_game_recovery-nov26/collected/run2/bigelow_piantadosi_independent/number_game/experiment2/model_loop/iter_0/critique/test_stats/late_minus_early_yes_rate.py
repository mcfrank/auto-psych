# name: late_minus_early_yes_rate
# description: Overall yes rate on trials with trial_index >= 225 minus trials with trial_index < 225; nonzero observed vs null (model has no time dependence) reveals drift in yes bias across the session.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    ti = df["trial_index"].astype(float).to_numpy()
    return rate(ti >= 225) - rate(ti < 225)
