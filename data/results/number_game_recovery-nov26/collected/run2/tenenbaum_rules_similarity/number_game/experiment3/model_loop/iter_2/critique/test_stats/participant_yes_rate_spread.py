# name: participant_yes_rate_spread
# description: SD across participants (rows blocked by session, a new participant starting at each trial_index 0) of the overall yes rate; observed above null means people differ in yes-bias more than the model's single shared decision criterion produces.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    r = df["rating"].values.astype(float)
    tot = np.bincount(pid, weights=r)
    cnt = np.bincount(pid)
    m = cnt > 0
    return float(np.std(tot[m] / cnt[m]))
