# name: participant_left_rate_sd
# description: Standard deviation across participants of each participant's proportion of Left choices; observed above null_mean means people have individual side biases that the model (no side-bias term) under-produces, below means less spread than predicted.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std(ddof=0))
