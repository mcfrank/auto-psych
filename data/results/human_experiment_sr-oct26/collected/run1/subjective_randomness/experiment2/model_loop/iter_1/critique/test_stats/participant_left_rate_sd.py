# name: participant_left_rate_sd
# description: Standard deviation across participants of their proportion of Left responses; observed above the null means individuals have personal side (position) biases that the model, which has no left/right bias term, under-produces.
def test_statistic(df):
    r = df.groupby("participant_id")["chose_left"].mean()
    return float(r.std(ddof=1))
