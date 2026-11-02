# name: trial_drift_residual
# description: Pearson correlation between trial_index and the rating's residual from its (set, target) cell mean; observed negative vs a null near zero means people say yes less often later in the session (a drift the stationary model lacks).
def test_statistic(df):
    r = df["rating"].astype(float)
    res = (r - df.groupby(["number_set", "target"])["rating"].transform("mean")).to_numpy()
    ti = df["trial_index"].astype(float).to_numpy()
    if np.std(res) == 0 or np.std(ti) == 0:
        return 0.0
    return float(np.corrcoef(ti, res)[0, 1])
