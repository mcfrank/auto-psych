# name: lag1_response_autocorrelation
# description: Correlation between a participant's rating on consecutive trials (pairs within the same session only); observed > null means sequential response perseveration/assimilation the trial-independent model does not produce (observed < null means alternation).
def test_statistic(df):
    t = df["trial_index"].values
    r = df["rating"].values.astype(float)
    ok = np.diff(t) == 1
    a, b = r[:-1][ok], r[1:][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
