# name: first_vs_repeat_set_exposure
# description: Yes rate on a participant's first trial with a given set (participant = run starting at trial_index 0) minus yes rate on later trials with that same set, non-example targets only; the model treats every trial independently, so observed above the null means generalization narrows with repeated exposure to a set (below = it broadens).
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    key = pd.Series(pid).astype(str) + "|" + df["number_set"].astype(str).to_numpy()
    k = pd.Series(key.to_numpy()).groupby(key.to_numpy()).cumcount().to_numpy()
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).astype(str).to_numpy()
    inset = np.array([tt in s.split(",") for tt, s in zip(t, sets)])
    y = df["rating"].to_numpy().astype(float)
    a, b = (k == 0) & ~inset, (k > 0) & ~inset
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
