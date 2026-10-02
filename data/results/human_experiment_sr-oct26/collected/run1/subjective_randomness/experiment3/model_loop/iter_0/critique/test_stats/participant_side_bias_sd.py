# name: participant_side_bias_sd
# description: Between-participant standard deviation of each person's left-choice rate; observed above null_mean means individuals have stable personal left/right response biases that the model (no side term) under-produces.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std())
