# name: participant_side_bias_variance
# description: Variance across participants of their proportion of Left choices; observed above null means individual side/response biases (or lapsing) that the model lacks and so under-produces.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().var(ddof=0))
