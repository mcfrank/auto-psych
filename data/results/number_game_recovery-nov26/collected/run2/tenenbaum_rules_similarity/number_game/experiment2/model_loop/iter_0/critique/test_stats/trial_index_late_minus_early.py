# name: trial_index_late_minus_early
# description: Yes rate on trials with trial_index >= 225 minus yes rate on trials < 225; the model has no drift, so observed far from the null means responding to identical stimuli changes over the session (negative = growing conservatism).
def test_statistic(df):
    ti = df["trial_index"].values
    r = df["rating"].values.astype(float)
    a = ti >= 225; b = ti < 225
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
