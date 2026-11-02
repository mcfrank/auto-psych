# name: participant_breadth_dispersion
# description: SD across participants (segmented by trial_index resets) of each participant's mean residual yes (rating minus the (set, target) cell's yes-rate across all participants); observed above the null means individuals differ in overall generalization breadth more than the single-parameter-set model produces, below means the model over-disperses participants.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    cell_mean = pd.Series(r).groupby(key.to_numpy()).transform("mean").to_numpy()
    res = pd.Series(r - cell_mean).groupby(pid).mean()
    return float(res.std(ddof=0))
