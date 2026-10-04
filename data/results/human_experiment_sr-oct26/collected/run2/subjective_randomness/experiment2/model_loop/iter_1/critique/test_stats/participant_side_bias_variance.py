# name: participant_side_bias_variance
# description: Variance across participants of their left-choice rate; observed above null means individuals have stable left/right response biases the model (no side bias) does not produce.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().var())
