# name: residual_lag1_autocorr
# description: Lag-1 correlation of response residuals (rating minus its (set,target) cell mean) between consecutive trials of the same session (trial_index increments by 1); observed above null means sequential response perseveration the trial-independent model lacks.
def test_statistic(df):
    r = df["rating"].values.astype(float)
    cm = df.groupby(["number_set", "target"])["rating"].transform("mean").values
    res = r - cm
    ti = df["trial_index"].values
    ok = np.diff(ti) == 1
    a = res[:-1][ok]; b = res[1:][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
