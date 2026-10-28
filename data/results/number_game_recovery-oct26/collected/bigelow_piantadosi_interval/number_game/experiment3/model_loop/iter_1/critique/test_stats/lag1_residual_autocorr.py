# name: lag1_residual_autocorr
# description: Lag-1 correlation of consecutive within-session residuals (rating minus number_set mean rating), sessions split where trial_index does not increase by 1; observed above null_mean means people perseverate on their previous answer beyond what the stimuli explain (model lacks sequential dependence).
def test_statistic(df):
    r = df["rating"].astype(float)
    res = (r - r.groupby(df["number_set"]).transform("mean")).to_numpy()
    ti = df["trial_index"].to_numpy()
    ok = np.diff(ti) == 1
    a = res[:-1][ok]
    b = res[1:][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
