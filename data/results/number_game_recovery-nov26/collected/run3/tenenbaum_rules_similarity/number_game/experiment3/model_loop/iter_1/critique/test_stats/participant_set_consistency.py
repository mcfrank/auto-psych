# name: participant_set_consistency
# description: SD across participant x set blocks (>=5 non-example trials) of the participant's yes rate on that set; observed above null means people commit to one hypothesis per set (responses within a set are more correlated than the model's shared-probability answers).
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    ex = sets.str.split(",")
    shown = np.array([str(a) in b for a, b in zip(t, ex)])
    g = pd.DataFrame({"pid": pid, "s": sets.values, "r": df["rating"].values})[~shown]
    a = g.groupby(["pid", "s"])["r"].agg(["size", "mean"])
    a = a[a["size"] >= 5]
    return float(a["mean"].std())
