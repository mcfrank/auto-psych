# name: new_set_first_trial_yes_gap
# description: Yes rate on non-example targets on the first trial after the set changes (or session start, trial_index 0) minus yes rate on subsequent trials with the same set as the previous trial; nonzero observed vs a null near the stimulus-driven value means a set-novelty effect the stationary model lacks (positive = more liberal on the first query about a new set).
def test_statistic(df):
    s = df["number_set"].astype(str)
    new = (s != s.shift(1)) | (df["trial_index"] == 0)
    ex = s.str.split(",")
    t = df["target"].astype(int).astype(str)
    isex = np.array([ti in e for ti, e in zip(t, ex)])
    r = df["rating"].to_numpy()
    a = new.to_numpy() & ~isex
    b = (~new.to_numpy()) & ~isex
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
