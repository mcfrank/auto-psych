# name: participant_side_bias_sd
# description: Standard deviation across participants of each participant's proportion of Left choices; observed above null_mean means people have individual left/right response biases that the model (no side-bias term) under-produces, below means less spread than predicted.
def test_statistic(df):
    r = df.groupby("participant_id")["chose_left"].mean()
    return float(r.std(ddof=0))
