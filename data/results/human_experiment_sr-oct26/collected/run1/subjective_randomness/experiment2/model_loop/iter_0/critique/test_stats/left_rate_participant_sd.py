# name: left_rate_participant_sd
# description: SD across participants of each person's proportion of Left responses; observed > null means people have personal side biases the model (no side term) underproduces.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std(ddof=0))
