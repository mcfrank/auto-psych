# name: trial_index_slope
# description: OLS slope of rating on trial_index scaled per 100 trials; a nonzero observed slope far from the null (which should be ~0 since the model has no time dependence) means humans' yes rate drifts over the session in a way the model cannot produce.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    vx = np.var(x)
    if vx == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (r - r.mean())) / vx * 100)
