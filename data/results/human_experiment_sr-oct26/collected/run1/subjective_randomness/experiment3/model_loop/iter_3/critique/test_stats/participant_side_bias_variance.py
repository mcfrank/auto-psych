# name: participant_side_bias_variance
# description: Variance across participants of each participant's proportion of Left choices; observed above null_mean means real people have personal left/right button biases the model (which has no side term) does not produce.
def test_statistic(df):
    return float(df["chose_left"].astype(float).groupby(df["participant_id"].to_numpy()).mean().var())
