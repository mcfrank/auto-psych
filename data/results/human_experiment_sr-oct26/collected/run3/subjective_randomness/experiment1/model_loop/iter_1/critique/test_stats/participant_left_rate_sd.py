# name: participant_left_rate_sd
# description: Standard deviation across participants of each person's proportion of Left choices; observed above null_mean means people differ in side bias more than the model's single shared side_bias allows.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std(ddof=0))
