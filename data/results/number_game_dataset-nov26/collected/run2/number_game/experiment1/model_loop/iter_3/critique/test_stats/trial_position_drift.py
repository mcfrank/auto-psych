# name: trial_position_drift
# description: Yes-rate on trials in the later half of the session (trial_index above median) minus the earlier half; the model has no trial-order term, so observed far from null means people's yes-bias drifts over the session (observed > null: more yes later; < null: fewer yes later).
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    med = np.median(ti)
    late = ti > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
