# name: participant_side_bias_variance
# description: Variance across participants of each participant's proportion of LEFT choices; observed above null_mean means individuals carry personal side (button) biases the model, which has no per-person side term, under-produces.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().var())
