# name: participant_left_rate_sd
# description: SD across participants of each person rate of choosing the left sequence; observed above null means left/right leans are more varied (some people click one side habitually) than the model side-bias distribution produces, below means the model over-produces side leans.
def test_statistic(df):
    per = df["chose_left"].groupby(df["participant_id"]).mean()
    return float(per.std()) if len(per) > 1 else 0.0
