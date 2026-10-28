# name: trial_index_drift
# description: Yes-rate in the first half of the session (trial_index below median) minus yes-rate in the second half; the model has no time dependence, so observed above the null means humans become more conservative over the session in a way the model does not produce (below: more liberal).
def test_statistic(df):
    ti = df["trial_index"].values.astype(float)
    r = df["rating"].values.astype(float)
    med = np.median(ti)
    early = ti < med
    late = ti >= med
    if early.sum() == 0 or late.sum() == 0:
        return 0.0
    return float(r[early].mean() - r[late].mean())
