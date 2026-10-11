# name: twin_uniform_singleton_choice_rate
# description: Rate of choosing the unique singleton object on utterance trials in the E9 twins experiment where the heard word is true of all objects.
def test_statistic(df):
    sub = df[(df["query"] == "utterance") & (df["experiment"] == "E9_twins") & (df["condition"] == "uniform")]
    if len(sub) == 0:
        return 0.0
    return float(np.mean(sub["choice"] == 0))
