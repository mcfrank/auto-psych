# name: participant_side_bias_spread
# description: Standard deviation across participants of each participant's proportion of Left choices; observed above the null means individuals have response-side biases (habitual Left or Right clicking) that the side-neutral model does not produce.
def test_statistic(df):
    r = df.groupby("participant_id")["chose_left"].mean()
    return float(r.std(ddof=0))
