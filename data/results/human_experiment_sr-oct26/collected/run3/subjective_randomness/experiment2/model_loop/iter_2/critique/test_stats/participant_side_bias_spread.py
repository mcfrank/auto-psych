# name: participant_side_bias_spread
# description: SD across participants of each person's proportion of Left choices; observed > null means the model (no side/response bias) under-produces person-level left/right response biases.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std())
