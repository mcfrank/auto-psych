# name: participant_side_bias_variance
# description: Variance across participants of their rate of choosing the LEFT sequence; observed above null_mean means people have idiosyncratic side (left/right) response biases the model lacks.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().var())
