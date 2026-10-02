# name: participant_side_bias_variance
# description: Variance across participants of each participant's proportion of Left choices; observed above null_mean means people carry personal left/right button biases that the model (no side term) under-produces.
def test_statistic(df):
    r = df.groupby("participant_id")["chose_left"].mean()
    return float(r.var(ddof=0))
