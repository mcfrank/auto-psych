# name: speakers_l2_target_choice_rate
# description: Rate of choosing the L2 target object on utterance trials in the speakers experiment where the competitor has an alternative unambiguous word.
def test_statistic(df):
    sub = df[(df["query"] == "utterance") & (df["experiment"] == "speakers")]
    if len(sub) == 0:
        return 0.0
    return float(np.mean(sub["choice"] == 1))
