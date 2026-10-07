# name: participant_left_rate_sd
# description: Across-participant SD of each person's left-choice rate; observed > null means individual side biases vary more than the model's single shared side_bias allows.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std())
